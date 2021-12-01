import numpy as np
from itertools import combinations
from datetime import datetime
import pandas as pd


class Queue(object):
    """Simulate a queue."""

    def __init__(self, arrival_rate, departure_rate, compatibility, values, hospital, pairs):
        """
        Parameters:
            arrival_rate: Exponential inter-arrival rate for pairs/donors
            departure_rate: Exponential departure rate (patient leaves system without matching)
            compatibility: Square matrix where M(i, j) denotes whether the donor of pair i can donate to patient of pair j.
            values: Matrix where V(i, j) denotes how much value donor i gives to patient j
            Altruistic donors are included in both matrices, with their corresponding "patient" having 0 for all donors.
            hospital: dataframe of the points each hospital has. Includes column 'points'.
            pairs: dataframe of patient-donor pairs in the system.
            Includes columns 'id', 'hospital_code', 'abo_donor', 'abo_patient', 'pra', and 'platform_code'.
        """
        self.current = []  # pool of people in the queue (arrived but not matched) - list of indexes
        self._arrivals = []  # list of arrival times
        self.departures = []  # list of departure times
        self.matches = []  # list of final matches (tuples of pair ids - not necessarily the indexes)
        self.num_transplants = 0
        self.num_arrivals = 0
        self.hospital = hospital

        self.setup_time = 0.0
        self.removal_time = 0.0

        self.current_clock = 0.0

        self._arrival_rate = arrival_rate
        self._departure_rate = departure_rate
        self._compatibility = compatibility
        self._values = values
        self._pairs = pairs
        self._ids = list(self._pairs.id) # list of ids with the same length of the compatibility and value matrices

    def generate_arrivals_departures(self, num_generate):
        """
        Generates all arrival and departure times for 'num_generate' pairs based on a poisson process with 'arrival_rate'
        and 'departure_rate'.
        Updates 'self._arrivals' with the list of arrival times and self._departures with the list of departure times.

        Parameters:
            num_generate: number of arrivals to generate
        """
        time = 0

        for i in range(num_generate):
            time += np.random.exponential(1.0 / self._arrival_rate)

            self._arrivals.append(time)

            depart = time + np.random.exponential(1.0 / self._departure_rate)

            self.departures.append(depart)

    def sample(self):
        """
        Sample (with replacement) from the pairs in order to generate arrivals to the pool.
        Returns the index of the pair given.
        """

        # ## only allow 2-3 or 3-2 (O-A and A-O)
        # abo_filtered = self._pairs.loc[((self._pairs.abo_donor == 'O') & (self._pairs.abo_patient == 'A')) | ((self._pairs.abo_donor == 'A') & (self._pairs.abo_patient == 'O'))]
        # return np.random.choice(abo_filtered.id)

        return np.random.randint(len(self._pairs))

    def remove_departures(self):
        """
        Remove pairs departing from the current pool (occurred before self._current_clock)
        """
        # Remove pairs in the active pool that have departure times less than current time
        for active_pair in reversed(range(len(self.current))):
            if self.departures[active_pair] <= self.current_clock:
                # Remove departure times from the list
                self.departures.pop(active_pair)
                self.current.pop(active_pair)

    def match(self, new_pair_index, both_directions=True, pairs=False):
        """
        Check to see if the new pair given can match to any of the pairs in the current pool.
        Returns a dictionary of indices that the pair can match with, along with the value associated with that match.

        Parameters:
            new_pair_index: Index of the patient-donor pair to check compatibility with current pairs.
            both_directions: Directionality of the match. By default, match both donors and patients for the two pairs.
            If False, the match is one-directional and checks for whether the donor can match to a patient (altruistic).
            pairs: Whether the new pair should match with two pairs and create a cycle of length 3.

        Returns:
            The index(es) of the pair(s) matched to the `new_pair_index`, followed by their match value.
            If there is no match, returns an empty dict.
        """
        matches = {}

        if not pairs:
            # Loop through current pairs in the pool.
            for potential_match in self.current:
                # Value of donating to this patient
                donor_value = self._compatibility[new_pair_index, potential_match] * self._values[new_pair_index, potential_match]
                # value of receiving this donor as a patient
                patient_value = both_directions * (self._compatibility[potential_match, new_pair_index] * self._values[potential_match, new_pair_index])

                # Both_directions: if both values > 0, that means M[i,j] and M[j,i] == 1, they are compatible.
                if (donor_value > 0) and ((patient_value > 0) == both_directions):
                    value = np.nan_to_num(patient_value) + donor_value
                    matches[potential_match] = value
        else:
            # Loop through combinations of pairs in the pool.
            for potential_pair in list(combinations(self.current, 2)):
                # Check if the new pair can be a donor to one of the pairs
                donor_cycle1 = self._compatibility[new_pair_index, potential_pair[0]] * self._values[new_pair_index, potential_pair[0]]
                donor_cycle2 = self._compatibility[potential_pair[0], potential_pair[1]] * self._values[potential_pair[0], potential_pair[1]]
                donor_cycle3 = self._compatibility[potential_pair[1], new_pair_index] * self._values[potential_pair[1], new_pair_index]
                cycle1 = (donor_cycle1 > 0) and (donor_cycle2 > 0) and (donor_cycle3 > 0)

                donor_cycle4 = self._compatibility[new_pair_index, potential_pair[1]] * self._values[new_pair_index, potential_pair[1]]
                donor_cycle5 = self._compatibility[potential_pair[1], potential_pair[0]] * self._values[potential_pair[1], potential_pair[0]]
                donor_cycle6 = self._compatibility[potential_pair[0], new_pair_index] * self._values[potential_pair[0], new_pair_index]
                cycle2 = (donor_cycle6 > 0) and (donor_cycle5 > 0) and (donor_cycle4 > 0)

                if cycle1:
                    value = donor_cycle1 + donor_cycle2 + donor_cycle3
                    matches[potential_pair] = value
                if cycle2:
                    value = donor_cycle4 + donor_cycle5 + donor_cycle6
                    matches[(potential_pair[1], potential_pair[0])] = value # list in order of the donor matches

        return matches

    def break_tie(self, potential_matches, use_points=True, pairs=False):
        """
        Takes in a dictionary of matches (key as index and value as match value) and returns the index with the maximum
        match value, tie-breaking by looking at the pair coming from the hospital with the greatest amount of points,
        or tie-breaking arbitrarily.
        If hospitals have the same number of points, can tie-break arbitrarily.

        Parameters:
            potential_matches: dictionary of indexes of pairs and their match values.
            use_points: if True, breaks ties between matches by using the hospital point system
            pairs: if True, three-way matching is on so the index is a tuple

        Returns:
            The index with the highest match value, tie-breaking with hospital points (if `use_points`=True).
        """
        # list of indexes that have the maximum match value
        best_match_index = [k for k, v in potential_matches.items() if v == max(potential_matches.values())]
        best_index = 0

        if use_points:
            # if there's more than one match in the list, need to tie-break
            if len(best_match_index) > 1:
                # if the index is a tuple, then get the index of the tuple with the highest hospital sum
                if pairs:
                    max_points = 0

                    # go over all indexes in the maximum match value list
                    for i in range(len(best_match_index)):
                        # hospital of the first and second pair in each index tuple
                        hospital0 = self._pairs.loc[best_match_index[i][0], 'hospital_code']
                        hospital1 = self._pairs.loc[best_match_index[i][1], 'hospital_code']

                        # get the sum of points from each pair in the index
                        hos_pts = self.hospital.loc[hospital0, 'points'] + self.hospital.loc[hospital1, 'points']

                        # if this current index has higher point value than the previous, make it the best index
                        if hos_pts > max_points:
                            best_index = i
                else:
                    # hospitals that the best matches correspond to
                    hospital_matches = self._pairs.loc[best_match_index, 'hospital_code']
                    # the hospital with the maximum points
                    hospital_max = self.hospital.loc[hospital_matches, 'points'].idxmax()
                    # index with the max points
                    best_match_index = hospital_matches.loc[hospital_matches == hospital_max].index.values

            return best_match_index[best_index]
        else:
            return best_match_index[np.random.randint(len(best_match_index))]

    def test_arrivals_departures(self):
        # Update current clock to the next arrival time
        self.current_clock = self._arrivals.pop(0)

        arrival_index = self.sample()
        self.num_arrivals += 1
        self.current.append(arrival_index)

        # Remove departures occurred during the previous period (less than current clock)
        self.remove_departures()


    def next_greedy_match(self, use_points=True, altruistic=True, three_way=True):
        """
        Match upon arrival (greedy). Add match to 'self.matches'
        Two-way cycles are always considered a match, but can vary if we find altruistic and three-ways as params.

        Parameters:
            use_points: if True, breaks ties between matches by using the hospital point system
            altruistic: if True, finds altruistic chains
            three_way: if True, finds three-way cycles
        """
        num_matches = len(self.matches)

        # Continue until we get a match
        while len(self.matches) == num_matches:
            arrival_index = self.sample()

            if (np.isnan(self._compatibility[:, arrival_index]).sum() == len(self._compatibility)) and (not altruistic):
                # throw away if altruistic donor and continue to next arrival
                continue

            # Update current clock to the next arrival time
            self.current_clock = self._arrivals.pop(0)

            self.num_arrivals += 1

            # Remove departures occurred during the previous period (less than current clock)
            self.remove_departures()

            # ########################### TESTING BELOW ###################################
            # match = 0
            # for potential_match in self.current:
            #     # Value of donating to this patient
            #     donor_value = self._compatibility[arrival_index, potential_match] * self._values[
            #         arrival_index, potential_match]
            #     # value of receiving this donor as a patient
            #     patient_value = self._compatibility[potential_match, arrival_index] * self._values[
            #         potential_match, arrival_index]
            #
            #     # Both_directions: if both values > 0, that means M[i,j] and M[j,i] == 1, they are compatible.
            #     if (donor_value > 0) and (patient_value > 0):
            #         self.matches.append((self._ids[arrival_index], self._ids[potential_match]))
            #
            #         self.hospital.loc[self._pairs.loc[arrival_index, 'hospital_code']] += self._pairs.loc[arrival_index, 'points']
            #         self.hospital.loc[self._pairs.loc[potential_match, 'hospital_code']] += self._pairs.loc[potential_match, 'points']
            #
            #         # remove the departure time of the pair just arriving to the pool and immediately got matched
            #         self.departures.pop(len(self.current))
            #
            #         removal_index = np.where(np.array(self.current) == potential_match)[0][0]
            #         self.current.pop(removal_index)
            #         self.departures.pop(removal_index)
            #
            #         self.num_transplants += 2
            #         match = 1
            #         break
            # if match == 0:
            #     self.current.append(arrival_index)

            ########################### TESTING ABOVE ###################################

            # Check to see if this new pair matches to anyone in the pool.
            # If the new pair is an altruistic donor (no patient attached in pair: patient column is all nan)
            if altruistic and (np.isnan(self._compatibility[:, arrival_index]).sum() == len(self._compatibility)):
                potential_matches = self.match(arrival_index, both_directions=False)

                loops = 0

                while potential_matches:
                    # Get the match with the highest value (break ties with hospital points)
                    best_match_index = self.break_tie(potential_matches, use_points)

                    # Add this match
                    self.matches.append((self._ids[arrival_index], self._ids[best_match_index]))

                    # Change the hospital's points
                    self.hospital.loc[self._pairs.loc[arrival_index, 'hospital_code']] += self._pairs.loc[arrival_index, 'points']
                    self.hospital.loc[self._pairs.loc[best_match_index, 'hospital_code']] += self._pairs.loc[best_match_index, 'points']

                    if loops == 0:
                        # remove the departure time of the donor just arriving to the pool that immediately got matched
                        self.departures.pop(len(self.current))

                    # Remove them from the pool and departure times (patient has been satisfied)
                    removal_index = np.where(np.array(self.current) == best_match_index)[0][0]
                    self.current.pop(removal_index)
                    self.departures.pop(removal_index)

                    # Loop to see if that donor paired with the patient can donate to someone in the pool
                    arrival_index = best_match_index
                    potential_matches = self.match(arrival_index, both_directions=False)
                    loops += 1
                    self.num_transplants += 1

                # No potential matches: add arrival to pool
                if loops == 0:
                    self.current.append(arrival_index)

            # If it's a regular patient-donor pair
            ## TODO: can clean up following code / consolidate cases
            else:
                # Compare cycle max to regular pair max
                regular = self.match(arrival_index, both_directions=True)

                if three_way:
                    # See if it can have a cycle
                    cycles = self.match(arrival_index, pairs=True)

                    # no cycle or regular matches
                    if ((not cycles) and (not regular)):
                        self.current.append(arrival_index)
                    # no regular pair matches or cycles's matches are larger in value
                    elif (not regular) or (cycles and (max(cycles.values()) >= max(regular.values()))):
                        best_cycle_index = self.break_tie(cycles, use_points, True)

                        self.matches.append((self._ids[arrival_index], self._ids[best_cycle_index[0]], self._ids[best_cycle_index[1]]))

                        # Change the hospital's points
                        self.hospital.loc[self._pairs.loc[arrival_index, 'hospital_code']] += self._pairs.loc[
                            arrival_index, 'points']

                        self.departures.pop(len(self.current))

                        for i in [0, 1]:
                            self.hospital.loc[self._pairs.loc[best_cycle_index[i], 'hospital_code']] += self._pairs.loc[
                                best_cycle_index[i], 'points']

                            removal_index = np.where(np.array(self.current) == best_cycle_index[i])[0][0]
                            self.current.pop(removal_index)
                            self.departures.pop(removal_index)

                        self.num_transplants += 3
                    # regular has a match but cycles does not, or that the max for regular is more than for cycles
                    elif (not cycles) or (regular and (max(cycles.values()) < max(regular.values()))):
                        best_regular_index = self.break_tie(regular, use_points)
                        self.matches.append((self._ids[arrival_index], self._ids[best_regular_index]))

                        # Change the hospital's points
                        self.hospital.loc[self._pairs.loc[arrival_index, 'hospital_code']] += self._pairs.loc[
                            arrival_index, 'points']
                        self.hospital.loc[self._pairs.loc[best_regular_index, 'hospital_code']] += self._pairs.loc[
                            best_regular_index, 'points']

                        self.departures.pop(len(self.current))
                        removal_index = np.where(np.array(self.current) == best_regular_index)[0][0]
                        self.current.pop(removal_index)
                        self.departures.pop(removal_index)

                        self.num_transplants += 2

                # just regular two-way matches
                elif regular:

                    best_regular_index = self.break_tie(regular, use_points)
                    self.matches.append((self._ids[arrival_index], self._ids[best_regular_index]))

                    # Change the hospital's points
                    self.hospital.loc[self._pairs.loc[arrival_index, 'hospital_code']] += self._pairs.loc[
                        arrival_index, 'points']
                    self.hospital.loc[self._pairs.loc[best_regular_index, 'hospital_code']] += self._pairs.loc[
                        best_regular_index, 'points']

                    self.departures.pop(len(self.current))
                    removal_index = np.where(np.array(self.current) == best_regular_index)[0][0]
                    self.current.pop(removal_index)
                    self.departures.pop(removal_index)

                    self.num_transplants += 2
                else:
                    self.current.append(arrival_index)


    def next_periodic_match(self, period):
        """
        Match at some periodic times. Add match to 'self.matches'

        Parameters:
            period: Time period to wait before matching with optimization / tie-breaking between pairs in the queue.
        """

        next_clock = self._current_clock + period

        # Add and remove people from the pool during the period
        while self._current_clock <= next_clock:
            first_arrival_time = self._arrivals[0]
            first_departure_time = self.departures[0]

            # arrival occurs first (or if they are concurrent, add an arrival and a departure)
            if (first_arrival_time <= first_departure_time) and (first_arrival_time <= next_clock):
                # sample with replacement from the pairs to add to the queue
                arrival_index = self.sample()
                self.current.append(arrival_index)

                # remove this arrival time from the list
                self._arrivals.pop(0)

            # departure occurs first (or if they are concurrent, add an arrival and a departure)
            if (first_departure_time <= first_arrival_time) and (first_departure_time <= next_clock):
                self.remove_departures()
                self.departures.pop(0)

            # update clock
            self._current_clock = min(first_arrival_time, first_departure_time, next_clock)

        # TODO: complete function with optimization of people in the current pool
        # Change matrices into graph format (write to csv) and call function from other repo






    # def generate_arrivals(self, num_arrivals):
    #     """
    #     Generates all arrivals times for 'num_arrivals' pairs based on a poisson process with 'arrival_rate'.
    #     Updates 'self._arrivals' with the list of arrival times.
    #
    #     Parameters:
    #         num_arrivals: Number of arrival times to generate
    #     """
    #     time = 0
    #     for i in range(num_arrivals):
    #         time += np.random.exponential(1.0 / self._arrival_rate)
    #
    #         self._arrivals.append(time)
    #
    # def generate_departures(self, num_departures):
    #     """
    #     Generates all departure times for 'num_departures' pairs based on a poisson process with 'departure_rate'.
    #     These are departures without matching.
    #
    #     TODO: parameter can depend on some characteristic of the pair, instead of deterministic departure rate
    #
    #     Parameters:
    #         num_departures: Number of departure times to generate
    #     """
    #     time = 0
    #     for i in range(num_departures):
    #         time += np.random.exponential(1.0 / self._departure_rate)
    #
    #         self._departures.append(time)

# def main():
#     dat = pd.read_csv('../Data/kematrix.csv', header=None)
#
#     ke = pd.read_csv('../Data/kedata.csv', header=None,
#                      names=['id', 'hospital_code', 'abo_donor', 'abo_patient', 'pra'])
#     ke[['hospital_code', 'platform_code']] = ke['hospital_code'].str.split('_', expand=True)
#
#     nkr = (ke.platform_code == "NKR")
#     ke_nkr = ke[nkr]
#
#     dat_nkr = dat[nkr].transpose()[nkr].transpose()
#
#     matrix = dat_nkr.to_numpy()
#
#     q = Queue(arrival_rate=10, departure_rate=1, compatibility=matrix, values=np.ones((len(matrix), len(matrix))),
#               ids=list(ke_nkr.id))
#     num_generate = 100000
#
#
# main()