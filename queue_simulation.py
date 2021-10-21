import numpy as np
from itertools import combinations
from datetime import datetime
import pandas as pd


class Queue(object):
    """Simulate a queue."""

    def __init__(self, arrival_rate, departure_rate, compatibility, values, ids):
        """
        Parameters:
            arrival_rate: Exponential inter-arrival rate for pairs/donors
            departure_rate: Exponential departure rate (patient leaves system without matching)
            compatibility: Square matrix where M(i, j) denotes whether the donor of pair i can donate to patient of pair j.
            values: Matrix where V(i, j) denotes how much value donor i gives to patient j
            Altruistic donors are included in both matrices, with their corresponding "patient" having 0 for all donors.
            ids: list of ids with the same length of the compatibility and value matrices
        """
        self.current = []  # pool of people in the queue (arrived but not matched) - list of indexes
        self._arrivals = []  # list of arrival times
        self._departures = []  # list of departure times
        self.matches = []  # list of final matches (tuples of pair ids - may be different from the indexes!)

        self._current_clock = 0.0

        self._arrival_rate = arrival_rate
        self._departure_rate = departure_rate
        self._compatibility = compatibility
        self._values = values
        self._ids = ids

    def generate_arrivals(self, num_arrivals):
        """
        Generates all arrivals times for 'num_arrivals' pairs based on a poisson process with 'arrival_rate'.
        Updates 'self._arrivals' with the list of arrival times.

        Parameters:
            num_arrivals: Number of arrival times to generate
        """
        time = 0
        for i in range(num_arrivals):
            time += np.random.exponential(1.0 / self._arrival_rate)

            self._arrivals.append(time)

    def generate_departures(self, num_departures):
        """
        Generates all departure times for 'num_departures' pairs based on a poisson process with 'departure_rate'.
        These are departures without matching.

        TODO: parameter can depend on some characteristic of the pair, instead of deterministic departure rate

        Parameters:
            num_departures: Number of departure times to generate
        """
        time = 0
        for i in range(num_departures):
            time += np.random.exponential(1.0 / self._departure_rate)

            self._departures.append(time)

    def sample(self):
        """
        Sample (with replacement) from the pairs in order to generate arrivals to the pool.
        Returns the index of the pair given.
        """
        return np.random.randint(len(self._compatibility))

    def remove_departures(self):
        """
        Remove pairs departing from the current pool (occurred before self._current_clock) *uniformly at random*.
        TODO: remove departures depending on some characteristic of the pair instead of randomly
        """
        # Number of departures (departure times that are less than current time)
        num_departures = sum(np.array(self._departures) <= self._current_clock)
        # Update departure list
        self._departures = self._departures[num_departures:]

        # Handle case where there are more departure times than people in the pool (because all times generated in advance)
        removals = min(num_departures, len(self.current))

        # Update current pool (remove random choice from the current pool)
        self.current = list(np.random.choice(self.current, size=len(self.current) - removals, replace=False))


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
                    value = patient_value + donor_value
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

    def next_greedy_match(self):
        """
        Match upon arrival (greedy). Add match to 'self.matches'
        """
        num_matches = len(self.matches)
        # num_arrivals_before_match = 0

        # Continue until we get a match
        while len(self.matches) == num_matches:

            # Update current clock to the next arrival time
            self._current_clock = self._arrivals.pop(0)

            arrival_index = self.sample()

            # Remove departures occurred during the previous period (less than current clock)
            self.remove_departures()

            # Check to see if this new pair matches to anyone in the pool.
            # If the new pair is an altruistic donor (no patient attached in pair: patient column is all nan)
            if np.isnan(self._compatibility[:, arrival_index]).sum() == len(self._compatibility):

                potential_matches = self.match(arrival_index, both_directions=False)
                loops = 0

                while potential_matches:
                    # Get the match with the highest value
                    best_match_index = max(potential_matches)
                    # Add this match
                    self.matches.append((self._ids[arrival_index], self._ids[best_match_index]))
                    # Remove them from the pool (patient has been satisfied)
                    self.current.remove(best_match_index)

                    # Loop to see if that donor paired with the patient can donate to someone in the pool
                    arrival_index = best_match_index
                    potential_matches = self.match(arrival_index, both_directions=False)
                    loops += 1

                # No potential matches: add arrival to pool
                if loops == 0:
                    self.current.append(arrival_index)

            # If it's a regular patient-donor pair
            else:
                # See if it can have a cycle
                # cycle_a = datetime.now()
                cycles = self.match(arrival_index, pairs=True)
                # cycle_b = datetime.now()
                # if (cycle_b - cycle_a).total_seconds() > 1:
                #     print("regular cycle time long")

                # Compare cycle max to regular pair max
                # regular_a = datetime.now()
                regular = self.match(arrival_index, both_directions=True)
                # regular_b = datetime.now()
                # if (regular_b - regular_a).total_seconds() > 1:
                #     print("regular pair time long")

                # no cycle or regular matches
                if ((not cycles) and (not regular)):
                    self.current.append(arrival_index)
                # no regular pair matches or cycles's matches are larger in value
                elif (not regular) or (cycles and (max(cycles.values()) >= max(regular.values()))):
                    best_cycle_index = max(cycles)
                    self.matches.append((self._ids[arrival_index], self._ids[best_cycle_index[0]], self._ids[best_cycle_index[1]]))
                    self.current.remove(best_cycle_index[0])
                    self.current.remove(best_cycle_index[1])
                # regular has a match but cycles does not, or that the max for regular is more than for cycles
                elif (not cycles) or (regular and (max(cycles.values()) < max(regular.values()))):
                    self.matches.append((self._ids[arrival_index], self._ids[max(regular)]))
                    self.current.remove(max(regular))

            # num_arrivals_before_match += 1
        # print("number of arrivals before match:", num_arrivals_before_match)

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
            first_departure_time = self._departures[0]

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
                self._departures.pop(0)

            # update clock
            self._current_clock = min(first_arrival_time, first_departure_time, next_clock)

        # TODO: complete function with optimization of people in the current pool
        # Change matrices into graph format (write to csv) and call function from other repo


# def main():
#     dat = pd.read_csv('../Data/kematrix.csv', header=None)
#
#     matrix = dat.to_numpy()
#
#     ke = pd.read_csv('../Data/kedata.csv', header=None,
#                      names=['id', 'hospital_code', 'abo_donor', 'abo_patient', 'pra'])
#     ke[['hospital_code', 'platform_code']] = ke['hospital_code'].str.split('_', expand=True)
#
#     q = Queue(arrival_rate=10, departure_rate=1, compatibility=matrix, values=np.ones((len(matrix), len(matrix))))
#
#     num_generate = 100000
#     q.generate_arrivals(num_generate)
#     q.generate_departures(num_generate)
#
#     while len(q.matches) < 10:
#         q.next_greedy_match()
#         print(q.matches)
#         print("current pool length", len(q.current))
#
#
# main()