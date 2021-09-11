import numpy as np


class Queue(object):
    """Simulate a queue."""

    def __init__(self, arrival_rate, departure_rate,
                 pairs, compatibility, values):
        """
        Parameters:
            arrival_rate: Exponential inter-arrival rate for pairs/donors
            departure_rate: Exponential departure rate (patient leaves system without matching)
            pairs: List of patient-donor pairs as tuples, including altruistic donors w/o associated patients
            compatibility: Square matrix where M(i, j) denotes whether the donor of pair i can donate to patient of pair j.
            values: Matrix where V(i, j) denotes how much value donor i gives to patient j
            Altruistic donors are included in both matrices, with their corresponding "patient" having 0 for all donors.
        """
        self._current = []  # pool of people in the queue (arrived but not matched) - list of indexes for the pairs
        self._arrivals = []  # list of arrival times
        self._departures = []  # list of departure times
        self.matches = []  # list of final matches (tuples of pair tuples)

        self._current_clock = 0.0

        self._arrival_rate = arrival_rate
        self._departure_rate = departure_rate
        self._pairs = pairs
        self._compatibility = compatibility
        self._values = values

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
        Sample (with replacement) from the list of pairs in order to generate arrivals to the pool.
        Returns the index from the list of pairs given.
        """
        return np.random.randint(len(self._pairs))

    def remove_departures(self):
        """
        Remove pairs departing from the current pool (occurred before self._current_clock) *uniformly at random*.
        TODO: remove departures depending on some characteristic of the pair instead of randomly
        """
        # Number of departures (departure times that are less than current time)
        num_departures = sum(np.array(self._departures) <= self._current_clock)

        # Handle case where there are more departure times than people in the pool (because all times generated in advance)
        removals = min(num_departures, len(self._current))

        # Update current pool
        self._current = list(np.random.choice(self._current, size = len(self._current) - removals))

    def match(self, new_pair_index):
        """
        Check to see if the new pair given can match to any of the pairs in the current pool.
        Tie-breaks by looking at the highest valued match (if there are multiple with the same value, arbitrarily pick the first).

        Parameters:
            new_pair_index: Index of the patient-donor pair to check compatibility with current pairs.

        Returns:
            The index of the pair matched to the `new_pair_index`.
            If there is no match, returns a garbage value of index -1.
        """
        highest_value = 0
        highest_match = -1

        ## TODO: Fix this function for altruistic donors
        # Loop through current pairs in the pool to look for the highest value.
        for potential_match in self._current:
            value1 = self._compatibility[potential_match, new_pair_index] * self._values[potential_match, new_pair_index]
            value2 = self._compatibility[new_pair_index, potential_match] * self._values[new_pair_index, potential_match]

            # If both values > 0, M[i,j] and M[j,i] == 1 and they are compatible. Check to see if this is higher than the current value.
            if (value1 > 0) and (value2 > 0) and (sum(value1, value2) > highest_value):
                # Update the highest value
                highest_value = sum(value1, value2)
                # Update the highest match index
                highest_match = potential_match

        return highest_match

    def next_greedy_match(self):
        """
        Match upon arrival (greedy). Add match to 'self.matches'
        """
        num_matches = len(self.matches)

        # Continue until we get a match
        while len(self.matches) == num_matches:

            # Update current clock to the next arrival time
            self._current_clock = self._arrivals.pop()

            arrival_index = self.sample()
            arrival = self._pairs[arrival_index]

            # Remove departures occurred during the previous period (less than current clock)
            self.remove_departures()

            # Check to see if this new pair matches to anyone in the pool.
            potential_match_index = self.match(arrival_index)
            if potential_match_index != -1:
                # If yes, add this match to self.matches
                self.matches.append((arrival, self._pairs[potential_match_index]))
            else:
                # If no, add arrival to the current pool waiting and continue
                self._current.append(arrival_index)

    def next_periodic_match(self, period):
        """
        Match at some periodic times. Add match to 'self.matches'

        Parameters:
            period: Time period to wait before matching with optimization / tie-breaking between pairs in the queue.
        """


def main():
    # Enter in data here
    q = Queue(arrival_rate=2, departure_rate=1,
              pairs=[(1, 2), (3, 4)],
              compatibility=np.array([[0, 1],
                                      [1, 0]]),
              values=np.array([[0, 20],
                               [10, 0]]))

    num_generate = 5
    q.generate_arrivals(num_generate)
    q.generate_departures(num_generate)

    # Run just once for the next greedy match
    q.next_greedy_match()

    # Returns the match
    return q.matches


if __name__ == '__main__':
    main()
