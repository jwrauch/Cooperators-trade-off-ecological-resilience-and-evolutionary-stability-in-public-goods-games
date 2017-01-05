import os as os
import numpy as np
from scipy import integrate, misc, optimize, linalg

DPATH = '/mnt/secondary/epgg/data/comp/'
VPATH = '/mnt/secondary/epgg/visuals/comp/'
RPATH = '/mnt/secondary/epgg/results/'
CPATH = '/home/jwrauch/code/python/epgg/comp/'

def create_environments(
        r = np.array([0.]), 
        N = np.array([0.]), 
        d = np.array([0.]),
        ):
    """Returns a list of environment arrays."""
    env = []
    for r_ in r:
        for N_ in N:
            for d_ in d:
                env.append([r_, N_, d_,])
    return np.array(env)

def create_random_environment(
        r = np.array([0.]), 
        N = np.array([0.]), 
        d = np.array([0.]),
        ):
    """Returns a random environment array based on the parameter constraints given."""

    env = np.empty(3)

    if r.shape == (1,): # a float in other words
        env[0] = r[0]
    elif r.shape == (2,):
        env[0] = np.random.rand()*np.diff(r) + r[0]
    else:
        raise EPGGError('Improper range for parameters given.')

    if N.shape == (1,):
        env[1] = N[0]
    elif N.shape == (2,):
        env[1] = np.random.randint(*N)
    else:
        raise EPGGError('Improper range for parameters given.')

    if d.shape == (1,):
        env[2] = d[0]
    elif d.shape == (2,):
        env[2] = np.random.rand()*np.diff(d) + d[0]
    else:
        raise EPGGError('Improper range for parameters given.')

    # If r was randomly chosen, it was chosen as an efficiency, r/N, from a range between 0
    # and 1.  Thus, it must be multiplied by N once that is known.
    if r.shape == (2,): 
        env[0] = env[0] * env[1]

    if env[0] < 2: # A known condition on r, it must be > 2 and < N (env[1])
        return create_random_environment(r, N, d) # Just try again
    return np.array(env)

def create_strategies(
        i = np.array([0.]), 
        k = np.array([0.]), 
        n = np.array([0.]),
        ):
    """Returns a list of strategy sets. NOTE: always and only competing against freeloaders"""
    strats_list = []
    for i_ in i:
        for k_ in k:
            for n_ in n:
                strats_list.append([[0., 0., 0.], [i_, k_, n_]])
    return np.array(strats_list)

def create_random_strategies(
        i = np.array([0.]), 
        k = np.array([0.]), 
        n = np.array([0.]),
        ):
    """Returns a random list of strategies based on the constraints given."""

    strats = np.array([np.zeros(3), np.empty(3)])

    if i.shape == (1,):
        strats[1][0] = i[0]
    elif i.shape == (2,):
        strats[1][0] = np.random.rand()*np.diff(i) + i[0]
    else:
        raise EPGGError('Improper range for parameters given.')

    if k.shape == (1,):
        strats[1][1] = k[0]
    elif k.shape == (2,):
        strats[1][1] = np.random.rand()*np.diff(k) + k[0]
    else:
        raise EPGGError('Improper range for parameters given.')

    if n.shape == (1,):
        strats[1][2] = n[0]
    elif n.shape == (2,):
        strats[1][2] = np.random.randint(*n)
    else:
        raise EPGGError('Improper range for parameters given.')

    return np.array(strats)

def create_initial_frequency_list(x0_lists):
    """Returns a list of intial frequency arrays to integrate.
    
    The x0_lists is a list of lists for each species in the community giving the initial frequency
    at which to start integrating.  The first list in x0_lists, x0_lists[0], is always freeloaders.
    
    """
    return np.dstack(np.meshgrid(*x0_lists)).reshape(-1, np.shape(x0_lists)[0])

def evolve_trajectories(community, env_list, strats_list, x0_list):
    """Returns a list of trajectories (lists) for each environment, strategy.
    
    The returned list is structured in a hiarchy of lists starting with environment, then strategy,
    and then initial condition/trajectory, so a trajectory =
    trajectories[enironment #][strategy #][initial condition #].
    
    """

    trajectories = []
    for e, env in enumerate(env_list):
        trajectories.append([])
        for s, strats in enumerate(strats_list):
            trajectories[e].append([])
            for i, x0 in enumerate(x0_list):

                community.set_environment(env)
                community.set_strategies(strats)

                trajectories[e][s].append(community.evolve(x0))

    return np.array(trajectories)

def calculate_fixed_points(community, env_list, strats_list):
    """Returns a list of fixed points (lists) for each environment and strategy.

    The returned list is structured in a hiarchy of lists starting with environment and then 
    strategy, So the list of fixed points for a community can be found as 
    fixed point list = fixed_points[environment #][strategy #].
    
    """

    fixed_points = []
    for e, env in enumerate(env_list):
        fixed_points.append([])
        for s, strats in enumerate(strats_list):
            # I don't need to append an array for strat as a seperate fixed point array will be
            # appended for each strategy already.

            community.set_environment(env)
            community.set_strategies(strats)

            points = community.fix_points()

            fixed_points[e].append(points)

    return np.array(fixed_points)

def random_fixed_points(community, nsets, env_ranges, strat_ranges):
    """Returns a list of fixed point arrays as well as corresponding env and strat lists.

    The returned lists should all be nsts long, and are all geneated one at a time, since it is
    unknown whether a parameter set is of any interest until after you've calculated it's
    stability.  The environments and strategies used to calculate the fixed points.  These lists
    can then be saved for later use.  

    The env_ranges and strat_ranges fed into the function are arrays of either a float specifying a
    specific value for the parameter, or an array specifying a range overwhich to choose the
    parameter.  

    """

    fix_pts_list = []
    env_list = []
    strats_list = []

    while len(fix_pts_list) < nsets:

        env = create_random_environment(*env_ranges)
        strats = create_random_strategies(*strat_ranges)

        community.set_environment(env)
        community.set_strategies(strats)

        points = community.fixed_points()

        if points[0][1].shape[0] == 1: # only extinction homogeneous point
            continue
        else:
            fix_pnts_list.append(points)
            env_list.append(env)
            strats_list.append(strats)

    return np.array(fix_pts_list), np.array(env_list), np.array(strats_list)

def calculate_optimal_unconditional_investment(community, env_list):
    """Returns an array of optimal investment for each env in env_list."""

    opt_i = np.zeros(env_list.shape[0])
    for e, env in enumerate(env_list):
        community.set_environment(env)
        opt_i[e] = community.critical_investments()[1]
    return opt_i

def calculate_ecological_resilience(community, env_list, strats_list, species=1):
    """Returns a list of ecological resiliences (lists) for each environment and strategy.

    The returned list is structured in a hiarchy of lists starting with environment and then 
    strategy, So the resiliences for a community can be found as 
    resilience = resilience[environment #][strategy #].
    
    """

    resilience = []
    for e, env in enumerate(env_list):
        resilience.append([])
        for s, strats in enumerate(strats_list):
            # I don't need to append an array for strat as a seperate fixed point array will be
            # appended for each strategy already.

            community.set_environment(env)
            community.set_strategies(strats)

            resilience[e].append(community.ecological_resilience(species))

    return np.array(resilience)

def calculate_evolutionary_stability_mutation(community, env_list, strats_list, species=1):
    """Returns a list of evolutionary stabilities (lists) for each environment and strategy.

    The returned list is structured in a hiarchy of lists starting with environment and then 
    strategy, So the stability for a community can be found as 
    stability = stability[environment #][strategy #].
    
    """

    stability = []
    for e, env in enumerate(env_list):
        stability.append([])
        for s, strats in enumerate(strats_list):
            # I don't need to append an array for strat as a seperate fixed point array will be
            # appended for each strategy already.

            community.set_environment(env)
            community.set_strategies(strats)

            stability[e].append(community.evolutionary_stability_mutation(species))

    return np.array(stability)

def random_resilience_and_stability(community, nsets, env_ranges, strat_ranges, mig=False):
    """Returns a list of resiliences and stabilites as well as corresponding env and strat lists.

    The returned lists should all be nsets long, and are all geneated one at a time, since it is
    unknown whether a parameter set is of any interest until after you've calculated it's
    stability.  The environments and strategies used to calculate the ecological resilience and
    evolution stability.  These lists can then be saved for later use.  

    The env_ranges and strat_ranges fed into the function are arrays of either a float specifying a
    specific value for the parameter, or an array specifying a range overwhich to choose the
    parameter. The mig parameter distinquishes between migrational and mutational stability. 

    """

    res_list = []
    stab_list = []
    fix_pt_list = []
    env_list = []
    strats_list = []

    while len(res_list) < nsets:

        env = create_random_environment(*env_ranges)
        strats = create_random_strategies(*strat_ranges)

        community.set_environment(env)
        community.set_strategies(strats)
        fix_pts = community.fix_points()

        res = community.ecological_resilience()
        if mig == True:
            stab = community.evolutionary_stability_migration()
        else:
            stab = community.evolutionary_stability_mutation()

        if np.isnan(res) or np.isnan(stab): 
            continue
        else:
            res_list.append(res)
            stab_list.append(stab)
            fix_pt_list.append(fix_pts)
            env_list.append(env)
            strats_list.append(strats)

    return (np.array(res_list), np.array(stab_list), np.array(fix_pt_list), np.array(env_list), 
            np.array(strats_list))

 

class Community:
    """An object for holding information on the species in the community and the environment.
    
    All community objects have:
    
        env - an array of environmental and game parameters
                        [efficiency, r, 0
                         max group size, N, 1
                         death rate, d, 2
                         ]

        strats - an array of arrays defining the strategy for each species in the community
                        [[investment amplitude, i, 0
                          investment theshold, k, 1
                          switching severity, n, 2] 
                          ]

        dirct - the directory tree for saving community informaiton with specified precision

    It should be noted that this is not a callable community, as in, it is a super class containing
    common functionality for real community subclasses, but should not be called itself.

    """


    def __init__(self): 
        """Initializes a community object."""
        pass

    def set_environment(self, environment):
        """Sets the environment attribute for the community."""
        if np.size(environment) != 3:
            raise EPGGError('Invalid Environment: '+str(environment))
        self.dirct = ''
        self.fix_pts = None
        self.env = np.array(environment)

    def set_strategies(self, strategies):
        """Sets the strategy attribute for the community."""
        if np.shape(strategies)[1] != 3:
            raise EPGGError('Invalid strategy: '+str(strategies))
        self.dirct = ''
        self.fix_pts = None
        self.strats = np.array(strategies)

    def set_directory(self, prec=2):
        """Sets the directory tree for saving community information with precision prec."""
        if not(os.uname()[1] == 'glados'):
            raise EPGGError('Must run on glados.')

        self.dirct = (self.__class__.__name__+
                '/r-'+str(self.env[0])+'_N-'+str(self.env[1])+'_d-'+str(self.env[2])+'/')
        if not(os.path.exists(DPATH+self.dirct)):
            os.mkdir(DPATH+self.dirct)

        for s, strat in enumerate(self.strats):
            self.dirct += ('spec-'+str(s)+
                    '_i-'+str(strat[0])+'_k-'+str(strat[1])+'_n-'+str(strat[2])+'_')
        self.dirct += '/'
        if not(os.path.exists(DPATH+self.dirct)):
            os.mkdir(DPATH+self.dirct)

    def save_trajectory(self, trajectory=np.array([[0.]]), prec=2):
        """Saves the trajectory to a location based on community parameter values."""

        self.set_directory(prec)

        tfile =  DPATH + self.dirct
        for s, s0 in enumerate(trajectory[0]):
            tfile += 'spec-'+str(s)+'_s0-'+str(s0)+'_'
        tfile += '.txt'
        np.savetxt(tfile, trajectory)

    def open_trajectory(self, x0=np.array([[1., 0.]]), prec=2):
        """Returns trajectory array inside  the file for given initial condition."""

        tfile =  DPATH + self.dirct
        for s, s0 in enumerate(x0):
            tfile += 'spec-'+str(s)+'_s0-'+str(s0)+'_'
        tfile += '.txt'
        return np.loadtxt(tfile)

    def fix_points(self):
        """Returns a list of fixed points for the community.  

        The fixed point list has a specific structure for all communities.  The returned list has
        three sublists; one for homogeneous fixed points, one for heterogeneous fixed points, and
        one for extinction fixed points.  The homogeneous fixed point list is structured such that
        it has a sublist for each species in the community, including freeloaders although they
        will almost always only have a homogeneous fixed point at extinction.  Each of these lists
        will be a list of points.  The other two sublists of fix_points, heterogeneous fixed points
        and extintion fixed points, are both simply lists of points.  

        """
        fix_pts = [[]]
        for sp, species in enumerate(self.strats):
            # Adds fix pt list to hom. pt list
            fix_pts[0].append(self.homogeneous_fixed_points(sp)) 
        fix_pts[0] = np.array(fix_pts[0]) # To make everything np arrays

        fix_pts.append(self.heterogeneous_fixed_points())

        fix_pts.append(self.extinction_fixed_points())

        self.fix_pts = np.array(fix_pts)
        return self.fix_pts

    def ecological_resilience(self, species=1):
        """Returns the ecological resilience of species in the community.
        
        Species is typically set to 1 as the cooperating species is typically the species of
        interest.
        
        """

        if self.fix_pts == None:
            fix_pts = self.fix_points()

        hom_pts = self.fix_pts[0][species]
        het_pts = self.fix_pts[1]

        if hom_pts.shape[0] == 1: # Only extinction
            return np.nan
        elif hom_pts[1][-1] < 0.: # unstable node or spiral for smaller hom pt
            return np.nan

        res = np.abs(np.log10(hom_pts[2][0]) - np.log10(hom_pts[1][0]))
        return res

    def evolutionary_stability_mutation(self, species=1):
        if self.fix_pts == None:
            fix_pts = self.fix_points()

        if np.isnan(self.ecological_resilience(species)):
            return np.nan

        hom_pts, het_pts, ext_pts = self.fix_pts

        if het_pts.shape[0] == 0: # no het pt
            if hom_pts[species][-1][-1] == 0.: # larger hom pt is a saddle, system unstable
                return np.nan
            elif hom_pts[species][-1][-1] == 1.: # larger hom pt is stable node, cooperators win
                return 1.   # Generally indicates no game, so return nan 
#                return np.nan
            else:
                raise EPGGError(
                        'Unknow evo stability: bod stablity: {0} for homogeneous pt'.format(
                        hom_pts[species][-1][-1]))

        elif het_pts.shape[0] == 1: # single het pt
            if (het_pts[0][-1] == 1.) or (het_pts[0][-1] == 2.): # stable node or spiral
                return het_pts[0][1] 
            elif (het_pts[0][-1] == -1.) or (het_pts[0][-1] == -2.): # unstable node or spiral
                return np.nan # always unstable system
            else:
                raise EPGGError(
                        'Unknow evo stability: bod stablity: {0} for heterogeneous pt'.format(
                        het_pts[0][-1]))

        elif het_pts.shape[0] == 2: # two het pts
            return 1.

        else: # only hd right now and so only one het pt possible
            raise EPGGError('Found too many heterogeneous pts: {0}'.format(het_pts))

    def evolutionary_stability_migration(self, species=1, tol=4):
        """Returns the evolutionary stability to migration for species."""

        if self.fix_pts == None:
            fix_pts = self.fix_points()

        if np.isnan(self.ecological_resilience(species)):
            return np.nan

        hom_pts, het_pts, ext_pts = self.fix_pts
#        zstr, qstr = hom_pts[1][-1][:-1]

#        mag = 1
#        qtest = 1. - 10**(-mag)
#        while mag < tol:
 #           x0 = np.array([zstr*(1.-qtest), zstr*qtest])
            
 #           x0 = np.array([zstr*(1.-qtest), zstr*qtest])
 #           try:
 #               traj = self.evolve(x0)
 #               tfate = traj[-1]
 #               zfate = np.sum(tfate)
 #               qfate = tfate[1] / zfate
 #           except EPGGError:
 #               qfate = 0. # Just not 1.
 #               qtest -= 10**(-mag)
 #               continue

 #           if  qfate > 1.-10**(-tol): # Returns to pure cooperator
 #               qtest -= 10**(-mag) # lower test point
 #           else:
 #               qtest += 10**(-mag) # return to a ess trajectory
 #               mag += 1 # increase magnitude of precision
 #               qtest -= 10**(-mag)

 #       if qtest + 10**(-mag) >= 1.:
 #           return np.nan
 #       return (qtest+10**(-mag)-10**(-mag+1) + qtest+10**(-mag)) / 2.


        if het_pts.shape[0] == 2: # two het points
            return 1. - het_pts[0][1]
        elif (het_pts.shape[0] == 0) and (hom_pts[species][-1][-1] > 0):
            return 1.
        else:
            return np.nan


class ClosedSum(Community):
    """This is a subclass of community for systems which can be summed to a closed form. 

    A closed form system means that the fitness functions for the species, an average of all
    possible payoffs weighted by the probability of recieving the payoff,  can be sumed to a closed
    form, and equation, without explicitly preforming the average.  These are specifically two
    species systems, implying that X[0] are always freeloaders and X[1] are always cooperators.
    This must be handled explicitally in fitness() because each will have it's own individual
    equation.  

    It should be noted that this is not a callable community, as in, it is a super class containing
    common functionality for real community subclasses, but should not be called itself.

    """

    def __init__(self): 
        """Initializes a community object."""
        pass

    def evolve(self, x0=np.array([0.])):
        """Returns an array of the community frequencies integrated in time."""

        if (self.env == None) or (self.strats == None):
            raise EPGGError('No environment or strategy set.')

        if np.sum(x0) > 1.:
            print 'x0 = ', x0
            raise EPGGError('Invalid Integrate call.')

        dt = 0.1
        max_step = 100
        t = np.linspace(0., max_step*dt, max_step)

        # define dynamical system
        dX_dt = lambda X, t: X * ((1.-np.sum(X))*self.fitness(X) - self.env[2])

        loop = 0 # the loop count is in case of limit cycles and to stop long integrations
        
        traj = integrate.odeint(dX_dt, x0, t)
        x0 = traj[-1]
        loop += 1
        while  (np.allclose(dX_dt(x0, 0), np.zeros(x0.size)) == False) and (loop < 1000):

            x0 = traj[-1]
            loop += 1
            traj = np.concatenate((traj, integrate.odeint(dX_dt, x0, t)[1:])) 
        
        if loop == 1000:
            print 'X = ', traj[-1]
            print 'dX_dt = ', dX_dt(traj[-1], loop)
            raise EPGGError('Integration failed to finish.')

        return traj

    def fitness(self, X):
        """Fitness returns the fitness for each species based on a closed form equation.

        The equation for each species is unique, but as there are only two and as freeloaders are
        always listed first, this can be handled explicitly.  

        """
        return np.array([self.freeloader_fitness(X), self.cooperator_fitness(X)])

    def extinction_fixed_points(self):
        """Returns a list of lists for the line of fixed points at zero population."""
        return np.array([[0., 0.05*i, 1.] for i in xrange(21)])

    def stability(self, pt, yx=False):
        """Returns a string of the stability of the fixed point."""

        if yx:
            Zeq = 1. - np.sum(pt)
            qeq = pt[1] / np.sum(pt)
        else:
            Zeq = 1. - pt[0]
            qeq = pt[1]


        jacob = self.jacobian([1.-Zeq, qeq])
        tr = np.trace(jacob)
        det = np.linalg.det(jacob)

        if det == 0:
            raise EPGGError('Boundary stability case for env: {0}, strats: {1}'.format(
                self.env, self.strats))
        elif det < 0:
            return 0. # 0 for saddle, neither stable nor unstable
        else:
            if tr < 0:
                if tr*tr/4. > det:
                    return 1. # +1 for a stable node
                elif tr*tr/4. < det:
                    return 2. # +2 for stable focus
                else:
                    raise EPGGError('Boundary stability case for env: {0}, strats: {1}'.format(
                        self.env, self.strats))
            elif tr > 0:
                if tr*tr/4. > det:
                    return -1. # -1 for an unstable node
                elif tr*tr/4. < det:
                    return -2. # -2 for unstable focus
                else:
                    raise EPGGError('Boundary stability case for env: {0}, strats: {1}'.format(
                        self.env, self.strats))

        

class HauertDoebeli(ClosedSum):
    """This is a subclass for unconditional cooperators, who invest a fixed amount every round.

    This system was first developed by Hauert and Doebeli in 2006 and is the basis for the whole
    project.  These cooperators invest a fixed amount every round of the game, paying a cost equal
    to the investment.  

    """

    def __init__(self): 
        """Initializes a community object."""

        self.env = None
        self.strats = None
        self.direct = ''
        self.fix_pts = None

    def freeloader_fitness(self, X):
        """Returns the fitness for a freeloader."""

        [r, N, d] = self.env
        z, [y, x] = 1.-np.sum(X), X
        i = self.strats[1][0]
        return r * i * x/(1.-z) * (1. - (1-z**N)/(N*(1.-z)))

    def cooperator_fitness(self, X):
        """Returns the fitness for a cooperator."""
        return self.freeloader_fitness(X) - self.fitness_difference(X)

    def fitness_difference(self, X):
        """Returns the difference between freeloader and cooperator fitnesses."""

        [r, N, d] = self.env
        z, [y, x] = 1.-np.sum(X), X 
        i = self.strats[1][0]
        return i * (1 + (r-1.)*z**(N-1) - (r/N)*(1-z**N)/(1.-z))

    def homogeneous_fixed_points(self, species=1):
        """Returns a list of lists for the homogeneous cooperating population fixed points.

        Fixed points are given as [ueq, qeq] = [1-Zeq, 1.] since we calculate Z and q=1. for
        homogeneous cooperating populations.

        Currently in the system, the freeloaders are incapable of surviving in a homogeneous
        populaions (aside from the trivial fixed point at extinction), and thus fixed points only 
        exist for the cooperating population.  These fixed points are easiest to calculate using
        tricks after transforming the system to u, total population density, and q, relative
        frequency of cooperators.  Thus, these tricks are highly dependant on the fitness
        equations, and are system specific.  For a reference, look in lab book at equations for 
        u_dot and q_dot.

        """
        if species == 0: # This is the index for the freeloading species
            return np.array([[0., 0., 1.]])

        [r, N, d] = self.env
        i = self.strats[1][0]

        dZ_dt = lambda Z: (d - (Z * (r-1) * i * (1-Z**(N-1.))))
        Zmin = optimize.fminbound(dZ_dt, 0., 1.)

        if dZ_dt(Zmin) > 0.:
            return np.array([[0., 1., 1.]]) # Only extinction

        Zeq = [optimize.brentq(dZ_dt, 0., Zmin), optimize.brentq(dZ_dt, Zmin, 1.)]
        return np.array([[0., 1., 1.]] + 
                [[1.-Z, 1., self.stability([0., 1.-Z], yx=True)] for Z in Zeq[::-1]])

    def heterogeneous_fixed_points(self):
        """Returns a list of lists for the heterogeneous population fixed points.

        The fixed points are given as [ueq, qeq].  These fixed points are easiest to calculate 
        using tricks after transforming the system to u, total population density, and q, relative
        frequency of cooperators.  Thus, these tricks are highly dependant on the fitness
        equations, and are system specific.  For a reference, look in lab book at equations for 
        u_dot and q_dot.

        For the Hauert and Doebeli system, there is at most one possible heterogeneous fixed point.

        """

        [r, N, d], i = self.env, self.strats[1][0]

        fit_diff = lambda Z: i * (1 + (r-1)*Z**(N-1) - (r/N)*(1-Z**N)/(1-Z))

        Zmin = optimize.fminbound(fit_diff, 0., 1.)
        if fit_diff(Zmin) > 0:
            raise EPGGError('Fitness difference never negative for parameters choosen. No game.')

        Zeq = optimize.brentq(fit_diff, 0., Zmin)

        qeq = (self.env[2] / 
                (self.strats[1][0] * (self.env[0]-1) * Zeq * (1-Zeq**(self.env[1]-1))))

        if qeq > 1:
            return np.array([])

        return np.array([[1-Zeq, qeq, self.stability([1.-Zeq, qeq])]])

    def jacobian(self, pt):
        """Returns the jacobian matrix at pt (always given in u, q space)."""

        [ueq, qeq] = pt
        Zeq = 1. - ueq

        [r, N, d] = self.env
        i = self.strats[1][0]

        u_dot = lambda Z, q: (1.-Z) * (Z*q*i*(r-1.)*(1.-Z**(N-1.)) - d)
        # Sort of awkwardly have to get the correct species frequency array for fitness difference.
        q_dot = lambda Z, q: -q*Z*self.fitness_difference([(1.-q)*(1.-Z), q*(1.-Z)])*(1.-q)
        
        du_du = lambda Z: -u_dot(Z, qeq)
        du_dq = lambda q: u_dot(Zeq, q)
        dq_du = lambda Z: -q_dot(Z, qeq)
        dq_dq = lambda q: q_dot(Zeq, q)

        return np.array(
                [[misc.derivative(du_du, Zeq, dx=1e-6), misc.derivative(dq_du, Zeq, dx=1e-6)], 
                [misc.derivative(du_dq, qeq, dx=1e-6), misc.derivative(dq_dq, qeq, dx=1e-6)]])

    def critical_investments(self):
        """Returns two critical investments which separate different phase spaces."""

        [r, N, d] = self.env

        fit_diff = lambda Z: (1 + (r-1)*Z**(N-1) - (r/N)*(1-Z**N)/(1-Z))
        Zmin = optimize.fminbound(fit_diff, 0., 1.)
        Zeq = optimize.brentq(fit_diff, 0., Zmin)

        i_c1 = d / (r-1.) / (N-1.) * N**(N/(N-1))

        i_c2 = d / ((r-1) * Zeq * (1-Zeq**(N-1)))

        return i_c1, i_c2


class GlobalPlastic(ClosedSum):
    """A class for cooperators who all invest the same amount which varies each turn.

    The key here is that all cooperators invest the same amount, so the investment is independent
    of the group construction for any particular cooperaotri, hence the global. An investment 
    strategy is typically a function, the primary one of interest being the hill function.  Each 
    strategy type, each strategy with a different function, will be a subclass of plastic 
    cooperators. They'll share a lot of the same functionality as many of the principles are the
    same, just subsututing a different investment functioin/strategy.  

    """

    def freeloader_fitness(self, X):
        """Returns the fitness for a freeloader."""

        [r, N, d] = self.env
        z, [y, x] = 1.-np.sum(X), X
        i = lambda x: self.investment(x)

        return r * i(x) * x/(1.-z) * (1. - (1-z**N)/(N*(1.-z)))

    def cooperator_fitness(self, X):
        """Returns the fitness for a cooperator."""
        return self.freeloader_fitness(X) - self.fitness_difference(X)

    def fitness_difference(self, X):
        """Returns the difference between freeloader and cooperator fitnesses."""

        [r, N, d] = self.env
        z, [y, x] = 1.-np.sum(X), X 
        i = lambda x: self.investment(x)
        
        return i(x) * (1 + (r-1.)*z**(N-1) - (r/N)*(1-z**N)/(1.-z))

    def homogeneous_fixed_points(self, species=0):
        """Returns a list of lists for the homogeneous cooperating population fixed points.

        Fixed points are given as [ueq, qeq] = [1-Zeq, 1.] since we calculate Z and q=1. for
        homogeneous cooperating populations.

        Currently in the system, the freeloaders are incapable of surviving in a homogeneous
        populaions (aside from the trivial fixed point at extinction), and thus fixed points only 
        exist for the cooperating population.  These fixed points are easiest to calculate using
        tricks after transforming the system to u, total population density, and q, relative
        frequency of cooperators.  Thus, these tricks are highly dependant on the fitness
        equations, and are system specific.  For a reference, look in lab book at equations for 
        u_dot and q_dot.

        """
        if species == 0: # This is the index for the freeloading species
            return np.array([[0., 0., 1.]])

        [r, N, d] = self.env
        i = lambda x: self.investment(x)

        dZ_dt = lambda Z: (d - (Z * (r-1) * i(1.-Z) * (1-Z**(N-1.))))
        Zmin = optimize.fminbound(dZ_dt, 0., 1.)

        if dZ_dt(Zmin) > 0.:
            return np.array([[0., 1., 1.]]) # Only extinction

        Zeq = [optimize.brentq(dZ_dt, 0., Zmin), optimize.brentq(dZ_dt, Zmin, 1.)]
        return np.array([[0., 1., 1.]] + 
                [[1.-Z, 1., self.stability([0., 1.-Z], yx=True)] for Z in Zeq[::-1]])

    def heterogeneous_fixed_points(self):
        """Returns a list of lists for the heterogeneous population fixed points.

        The fixed points are given as [ueq, qeq].  These fixed points are easiest to calculate 
        using tricks after transforming the system to u, total population density, and q, relative
        frequency of cooperators.  Thus, these tricks are highly dependant on the fitness
        equations, and are system specific.  For a reference, look in lab book at equations for 
        u_dot and q_dot.

        For the Hauert and Doebeli system, there is at most one possible heterogeneous fixed point.

        """

        [r, N, d] = self.env
        i = lambda x: self.investment(x)

        # Although this is multiplied by the investment function in the class function, the
        # investment function never crosses zero, thus not affecting where the zero of the function
        # is.  That makes this a 1 dimensional problem which is much easier.
        fit_diff = lambda Z: 1 + (r-1)*Z**(N-1) - (r/N)*(1-Z**N)/(1-Z)

        Zmin = optimize.fminbound(fit_diff, 0., 1.)
        if fit_diff(Zmin) > 0:
            raise EPGGError('Fitness difference never negative for parameters choosen. No game.')

        Zeq = optimize.brentq(fit_diff, 0., Zmin)

        q_null = lambda q: Zeq * q * i((1.-Zeq)*q) * (r-1) * (1.-Zeq**(N-1.)) - d

        if q_null(0.) * q_null(1.) < 0.:
            qeq = np.array([optimize.brentq(q_null, 0., 1.)])
        elif q_null(0.) + q_null(1.) > 0.:
            return np.array([])
        else:
            func = lambda q: -q_null(q)
            qmax = optimize.fminbound(func, 0., 1.)
            if q_null(0.) * q_null(qmax) < 0:
                qeq = np.array([optimize.brentq(q_null, 0., qmax), 
                    optimize.brentq(q_null, qmax, 1.)])[::-1] # reverses order so unstable pt first
            else:
                return np.array([])

        return np.array([[1-Zeq, q, self.stability([1.-Zeq, q])] for q in qeq])

    def jacobian(self, pt):
        """Returns the jacobian matrix at pt (always given in u, q space)."""

        [ueq, qeq] = pt
        Zeq = 1. - ueq

        [r, N, d] = self.env
        i = lambda x: self.investment(x)

        u_dot = lambda Z, q: (1.-Z) * (Z * q * i((1.-Z)*q) * (r-1.) * (1.-Z**(N-1.)) - d) # u*q = x
        # Sort of awkwardly have to get the correct species frequency array for fitness difference.
        q_dot = lambda Z, q: -q*Z*self.fitness_difference([(1.-q)*(1.-Z), q*(1.-Z)])*(1.-q)
        
        du_du = lambda Z: -u_dot(Z, qeq)
        du_dq = lambda q: u_dot(Zeq, q)
        dq_du = lambda Z: -q_dot(Z, qeq)
        dq_dq = lambda q: q_dot(Zeq, q)

        return np.array(
                [[misc.derivative(du_du, Zeq, dx=1e-6), misc.derivative(dq_du, Zeq, dx=1e-6)], 
                [misc.derivative(du_dq, qeq, dx=1e-6), misc.derivative(dq_dq, qeq, dx=1e-6)]])

class GlobalHillCooperators(GlobalPlastic):
    """This is a class the global plastic cooperators who follow a hill function type strategy."""

    def __init__(self): 
        """Initializes a community object."""

        self.env = None
        self.strats = None
        self.direct = ''
        self.fix_pts = None

    def investment(self, x):
        """Returnst the investment based on the parameters in strat. """

        [i, k, n] = self.strats[1]
        return i / (1. + (x/k)**n)



class OpenSum(Community):
    """This is a subclass of community for system which can not be summed to closed form.

    For open systems, the fitness function for each species must be calculated by averaging all of
    the potential payoffs a species can recieve weighted by the probability of recieving that
    payoff.  There is no closed for equation short-cut.

    """

    def __init__(self, environment, strategies, precision=2): 
        """Initializes a community object."""
        pass

    def all_groups(N, k):
        """Returns an array of lists, one list for each possible group composition.

        This function splits the N possible spots in an interation group amoungst the k species,
        such that each group is an array of giving the player count for each species in the group.
        Then an array of all these group arrays is returned. With only three species thus far, the
        group array is structured: 

        [empty, freeriders, cooperators]

        """
        # Code taken from Greg Kuperberg on mathoverflow.net answering "Uniquely generate all
        # permutations of three digits that sum  to a particular value?"
        if (k < 0) or (N < 0): raise EPGGError("Incorrect group construction")

        if not k: return [[0]*N]
        if not N: return []
        if N == 1: return [[k]]

        return ([[0] + val for val in all_groups(N-1, k)] + [[val[0]+1] + val[1:] for val in
            all_groups(N, k-1)])

    def all_payoffs(groups, strategies, environment):
        """Returns the payoff each species recieves if found in group."""
        return np.array([environment[0] * investment(grp, strategies) / np.sum(grp[1:]) 
            for grp in groups[:-1]] + [0.]) # I manually add the last group, the group of all
                                                  # empty space, so as to avoid dividing by 0

    def investment(group, strategies):
        """Returns the total investment the group makes into the public good based on strat.
        
        group is an array of length len(strat)+1, whose elements signify how many of that species can be
        found in group.  Each member then constributes strategy, based on the group composition and
        it's own strat.
        
        """
        return np.sum([k * strategy(strat, group) for strat, k in zip(strategies, group[1:])])

    def multinomial(N, ks):
        """Returns the multinomial coefficient for N choose ks."""
        return misc.factorial(N) / np.prod(misc.factorial(ks))

    def fitness(frequencies, strategies, groups, payoffs, environment):
        """Returns the fitness for each species given its strat, and the chances of group.

        This return an array of length strat giving the fitness of each species in the community by
        averaging over the payoff from each group whose probability is determined by the frequency of
        each species.

        """
        return np.array([np.sum([multinomial(environment[1], grp) * # number of ways to make group
            np.prod(np.power(np.append([1.-np.sum(frequencies)], frequencies), grp)) * # prob of group
            (pay - strategy(strat, grp)) for grp, pay in zip(groups, payoffs)]) 
            for strat in strategies])
                
    def strategy(strat, group):
        """Returns the investment strat makes given group."""

        if strat.size != 3:
            raise EPGGError('Invalid strategy:', str(strat))
        elif np.sum(strat[1:]) == 0.: # unconditional cooperators: only i is non-zero
            return strat[0]
        else: # hill cooperators: only other strategy so far
            X_l = np.sum(group[2:]) / np.sum(group) # number of coop / total group size
            return strat[0] / (1. + np.power(X_l / strat[1], strat[2]))

    def evolve(self, x0=np.array([0.]), t=np.linspace(0., 100000, 1000000)):
        """Returns an array of the community frequencies integrated in time."""

        if np.sum(x0) > 1.:
            raise EPGGError('Invalid Integrate call.')

        # all possible groups that can randomly form
        groups = np.array(all_groups(np.size(x0)+1, self.env[1])) # +1 for empty space

        # all payoffs for all possible groups
        payoffs = all_payoffs(groups, self.strats, self.env)

        # define dynamical system
        dX_dt = lambda X, t, strategies, groups, payoffs, environment: [
                x * ((1.-np.sum(X))*fitness(X, strategies, groups, payoffs, environment)[i] - 
                environment[2]) for i, x in enumerate(X)]

        return integrate.odeint(dX_dt, x0, t, 
            (self.strats, groups, payoffs, self.env))

    def homogeneous_fixed_points(self):
        """Returns an array of tuples for the steady state frequencies of the community."""
        
        min_res = optimize.minimize(func, 0.5, bounds=np.array([0., 1.]))
        if not(min_res.success):
            raise EPGGError('Minimizer failed.')

        max_res = optimize.minimize(func, 0.5, bounds=np.array([0., 1.]))
        if not(max_res.success):
            raise EPGGError('Mazimixe failed.')

        max_and_min = np.unique(np.append(min_res.x, max_res.x))

class EPGGError(Exception):
    """Exception used to handle unstable systems."""
    def __init__(self, message):
        Exception.__init__(self, message)

