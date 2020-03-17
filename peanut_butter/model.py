from scipy.integrate import solve_lvp


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
            * beta[i, j] - the transmission rate of someone in region i with someone in region j
            * gamma - the recovery rate

        This model adds spatial compartments to a classic SIR model, so that
        S, I and R are multi-dimensional vectors. Unlike other spatially
        separated SIR models, the population doesn't move i.e. the total population
        in each region remains fixed, they can merely infect across region boundaries.
        The reasoning here is that the model operates on a time scale of a day
        (t has units of days) and each agent that leaves their home region returns
        to their home region by the end of the day - so they may carry the
        infection into a neighbouring region, come into contact with a susceptible
        and then return home, or alternatively, a susceptible may visit a
        neighbouring region, come into contact with an infected agent and then
        return home. Either process transmits the infection across boundary lines.
        """
        S, I, R = indexer.unpack(y)

        dS = -np.dot(beta,  I) * S / N
        dI = np.dot(beta, I) * S / N - gamma * I
        dR = gamma * I

        return indexer.pack(dS, dI, dR)

    y0 = Indexer.pack(S0, I0, R0)
    return solve_ivp(
        f,
        (times[0], times[-1]),
        y0,
        t_eval=times
    )
