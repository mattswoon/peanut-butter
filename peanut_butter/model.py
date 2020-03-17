

def model(S0, I0, R0, times, beta, N, gamma, indexer):
    """
    The model is described by

        y' = f(t, y)
    """

    def f(t, y):
        """
        RHS of a multi-variate version of the an SIR model.

        Each variable is a vector of the number of agents in that compartment
        indexed over a number of regions.

            * S[i] - the susceptible population in the ith region
            * I[i] - the infected population in the ith region
            * R[i] - the recovered population in the ith region
            * beta[i, j] - the contact rate of someone in region i with someone in region j
            * gamma - the recovery rate
        """
        S, I, R = indexer.unpack(y)

        dS = -np.dot(beta,  I * S) / N
        dI = np.dot(beta, I * S) / N - gamma * I
        dR = gamma * I

        return indexer.pack(dS, dI, dR)

    y0 = Indexer.pack(S0, I0, R0)
    return solve_ivp(
        f,
        (times[0], times[-1]),
        y0,
        t_eval=times
    )
