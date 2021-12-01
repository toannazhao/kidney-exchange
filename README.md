# Kidney Exchange

Creates a simulation of the kidney exchange market according to a Poisson process for patient-donor pairs (and altruistic donors) with optimal matching (greedy or periodic).

The input data needed is as follows:
* Compatibility matrix (square matrix) where M(i, j) denotes whether the donor of pair i can donate to patient of pair j.
* Matrix of values (square matrix) where V(i, j) denotes how much value donor i gives to patient j.
Altruistic donors are included in both matrices, with their corresponding "patient" having NaN for all donors.
* Dataframe of the points each hospital has to begin with (can initialize with all zeros): the index is the name of the hospital, and there is one column of 'points'.
* Dataframe of patient-donor pairs in the system, including their 'id', 'hospital_code', 'abo_donor', 'abo_patient', 'pra', and 'platform_code'.

To set up the simulation, arrival and departure times are generated for each pair that arrives before running anything. These times are stored in lists.

Then, the (greedy) simulation is as follows:

1. Until there is a match, sample an arrival to the system from the list of pairs (with replacement) and update the current time.
2. Remove the pairs that have departed from the system prior to the current time.
3. For the current pair that just arrived, look for a match in the current pool (pairs in the system that are waiting and have not been matched yet) that gives the highest value.
	a. Altruistic donors can create chains. We can also look for three-way matches.
	b. We can break ties in values either arbitrarily or by using hospital points.
4. If the current pair finds a match, remove them from the pool. If not, sample another pair to join the pool and repeat the steps.

The simulation terminates when it finds a match. To run it for multiple matches, loop over the function for some goal number of matches.
The function tracks the current pool, the list of final matches (tuples of pair ids), the total number of transplants, the total number of arrivals, and hospital points.

