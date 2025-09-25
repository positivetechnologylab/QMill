
import matplotlib.colors as mcolors

### COLORS ###------------------------------------------------------------------

firstColors = list(mcolors.BASE_COLORS)
colors = list(mcolors.TABLEAU_COLORS) + firstColors[:-1]



### DIST RESULTS FUNCTIONS ###--------------------------------------------------


def TVDPlot(ansatzList, dist):

    '''
    Similar to tolerancePlot, but only takes in an ansatz list and dist. Might 
    modify to include multiple dists per plot?
    '''

    names = list(map(lambda x : f'{x.name}, {x.depth}', ansatzList))

    fig, ax = plt.subplots()

    results = []

    for ansatz in ansatzList:
        result = fullDist(ansatz, dist, getTVD = True)
        results.append(result)
    
    ax.plot(names, results, 'o--', color = colors[0])

    plt.xlabel('Ansatz')
    plt.ylabel('TVD')
    plt.savefig(f'test_{dist.name}.png')


    


ansatzList = [Five(3,1), Six(3,1), Thirteen(3,1), Fourteen(3,1), 
              Five(3,2), Six(3,2), Thirteen(3,2), Fourteen(3,2)]
D1AnsatzList = [Five(3,1), Six(3,1), Thirteen(3,1), Fourteen(3,1)]


TVDPlot(ansatzList, Normal(200))
TVDPlot(ansatzList, Uniform(200))
TVDPlot(ansatzList, WeibullRight(200))
TVDPlot(D1AnsatzList, WeibullLeft(200))


# fullDist(Five(3,2), WeibullRight(200))