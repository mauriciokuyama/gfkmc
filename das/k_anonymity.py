# adapted from:
# https://www.mdai.cat/code/v20230316/prog.vectors.matrices.web.txt
# https://www.mdai.cat/code/v20230316/prog.sdc.web.txt

import numpy as np
import pandas as pd
import timeit


# def fNorm(v1, v2):
#     # return np.sqrt(np.sum((v1 - v2) ** 2))
#     return np.linalg.norm(v1 - v2)

def fNorm2(v1, v2):
    return np.sum((v1 - v2) ** 2)


def selectFv(lofe, lofv, f):
    # lofe = np.array(lofe)
    lofv = np.array(lofv)

    mask = f(lofv)
    res = lofe[mask]

    return res

def matrixColumnFunction(mat, f):
    numRows = len(mat)
    res = []
    if numRows != 0:
        numCols = len(mat[0])
        # print(numCols)
        for i in range(numCols):
            res.append(f(list(map(lambda x: x[i], mat))))
    return res

def matrixColumnMeans(mat):
    numRows = len(mat)
    res = np.array([])

    if numRows != 0:
        res = np.mean(mat, axis=0)

    return res

def matrixColumnSD(mat):
    colMeans = np.mean(mat, axis=0)
    res = np.sqrt(np.sum((mat - colMeans) ** 2, axis=0) / (mat.shape[0] - 1))
    return res

def farthestRow(db, vect, vDistance=fNorm2):
    selectedRow = np.argmax([vDistance(vect, row) for row in db])
    return selectedRow

def mdav(df: pd.DataFrame, QIs: list, k: int) -> pd.DataFrame:
    start = timeit.default_timer()

    new_df = df.copy()
    db = new_df[QIs].to_numpy()
    assignedCl = mdavCl(db, k)
    values = set(assignedCl)
    centroids = []

    max_value = db.max(axis=0)
    min_value = db.min(axis=0)
    ncp = 0

    for cl in values:
        rowsCl = selectFv(db, assignedCl, lambda x: x == cl)
        mat = np.array(rowsCl)

        max_row = mat.max(axis=0)
        min_row = mat.min(axis=0)

        # print(mat)
        # print(max_row)
        # print(min_row)

        ncp += sum(((max_row - min_row) * mat.shape[0]) / (max_value - min_value))

        meanRow = matrixColumnMeans(mat)
        centroids.append(meanRow)

    newDb = [centroids[assignedCl[i]] for i in range(0, len(db))]
    new_df[QIs] = newDb

    n_registers = new_df.shape[0]
    n_attributes = new_df.shape[1]
    ncp = ncp / (n_registers * n_attributes)
    print(f'ncp={ncp}')

    stop = timeit.default_timer()
    execution_time = stop - start
    print(f"Execution Time: {execution_time}")

    return new_df


def mdavCl(db, k):
    ## if (len(db)<2*k):
    ##     cl = [0]*len(db)
    ## else:
    assignedClass = [-1] * len(db)
    C = []
    clNumber = -1
    nPendingElements = len(db)
    while nPendingElements >= 3 * k:
        unassigned = selectFv(db, assignedClass, lambda x: x == -1)
        meanX = matrixColumnMeans(unassigned)
        xr = farthestRow(unassigned, meanX)
        xs = farthestRow(unassigned, unassigned[xr])
        # print("xr="+str(xr))
        # print("xs="+str(xs))
        toO, dbO, indexCr = kClosestToVect(db, assignedClass, unassigned[xr], k)
        clNumber = clNumber + 1
        assignedClass = updateCl(indexCr, clNumber, assignedClass)
        toO, dbO, indexCs = kClosestToVect(db, assignedClass, unassigned[xs], k)
        clNumber = clNumber + 1
        assignedClass = updateCl(indexCs, clNumber, assignedClass)
        nPendingElements = nPendingElements - 2 * k
    if nPendingElements >= 2 * k:
        unassigned = selectFv(db, assignedClass, lambda x: x == -1)
        meanX = matrixColumnMeans(unassigned)
        xr = farthestRow(unassigned, meanX)
        # print("xr="+str(xr))
        toO, dbO, indexCr = kClosestToVect(db, assignedClass, unassigned[xr], k)
        clNumber = clNumber + 1
        assignedClass = updateCl(indexCr, clNumber, assignedClass)
        nPendingElements = nPendingElements - k
    clNumber = clNumber + 1
    assignedClass = remainingCl(clNumber, assignedClass)
    return assignedClass


# Function:
#   assign unassigned positions to class clNumber
def remainingCl(clNumber, assignedClass):
    for i in range(0, len(assignedClass)):
        if assignedClass[i] == -1:
            assignedClass[i] = clNumber
    return assignedClass


## remainingCl(500,[-1,-1,2,3,4,5,6,-1,8,9,0])


# Function:
#    add to all indices in indexCs the class identifier: clNumber
def updateCl(indexCls, clNumber, assignedClass):
    for i in range(0, len(indexCls)):
        assignedClass[indexCls[i]] = clNumber
    return assignedClass


# Function:
#   select the nearest k rows in Db and return them with the corresponding assignments
def kClosestToVect(db, assignments, vect, k):
    toOrder = []
    dbOrder = []
    indexDb = []
    i = 0
    addedRows = 0
    while addedRows < k:
        if assignments[i] == -1:
            # print("addedRows="+str(addedRows)+":(v,db[i="+str(i)+"])="+str(vect)+","+str(db[i]))
            toOrder.append(fNorm2(vect, db[i]))
            dbOrder.append(db[i])
            indexDb.append(i)
            addedRows = addedRows + 1
        i = i + 1
    toOrder, dbOrder, indexDb = orderWithList(toOrder, dbOrder, indexDb)
    while i < len(db):
        if assignments[i] == -1:
            d = fNorm2(vect, db[i])
            if d < toOrder[k - 1]:
                toOrder, dbOrder, indexDb = addOrderedWithList(
                    toOrder, dbOrder, indexDb, d, db[i], i
                )
        i = i + 1
    return (toOrder, dbOrder, indexDb)


## Function:
##    Order in increasing order using a vector
def orderWithList(toOrder, db, indexDb):
    indices = np.argsort(toOrder)
    print(indices)
    toOrder = toOrder[indices]
    db = db[indices]
    indexDb = indexDb[indices]
    return (toOrder, db, indexDb)


## Function:
##   add a tuple (vect, distance, index) in an already ordered list with distances
##   in increasing order
def addOrderedWithList(toOrder, dbOrder, indexDb, d, vect, indexVect):
    l = len(toOrder)
    if toOrder[l - 1] > d:
        i = 0
        while toOrder[i] <= d:
            i = i + 1
        j = l - 1
        ## toOrder[i]>d,    put at toOrder[i]=d
        while j > i:
            toOrder[j] = toOrder[j - 1]
            dbOrder[j] = dbOrder[j - 1]
            indexDb[j] = indexDb[j - 1]
            j = j - 1
        toOrder[i] = d
        dbOrder[i] = vect
        indexDb[i] = indexVect
    return (toOrder, dbOrder, indexDb)


## Information Loss: IL ------------------------------------------


## Function:
##   a function to compute information loss in terms of
##     1) difference of means
##     2) difference of standard deviations
##     3) maximum difference between the two matrices
def sdcIL_stats(df_anon: pd.DataFrame, df_orig: pd.DataFrame, QIs: list):
    # x = [[i] for i in df_anon[QI]]
    # xp = [[i] for i in df_orig[QI]]
    x = df_anon[QIs].to_numpy()
    xp = df_orig[QIs].to_numpy()
    meanX = matrixColumnMeans(x)
    sdX = matrixColumnSD(x)
    meanXp = matrixColumnMeans(xp)
    sdXp = matrixColumnSD(xp)
    dMean = fNorm2(meanX, meanXp)
    dSD = fNorm2(sdX, sdXp)
    dMax = max(list(map(lambda x: max(x), np.subtract(x, xp))))
    return (dMean, dSD, dMax)



## DR: Disclosure Risk ------------------------------------------


#   sdcRecordLinkage (mdav(fRows, 3), fRows)
# def sdcRecordLinkage(df_anon: pd.DataFrame, df_orig: pd.DataFrame, QIs: list):
#     # x = df_anon[QI].to_list()
#     # xp = df_orig[QI].to_list()
#     # x = [[i] for i in df_anon[QI]]
#     x = df_anon[QIs].to_numpy()
#     xp = df_orig[QIs].to_numpy()
#     match = 0
#     for i in range(0, len(x)):
#         iMin = 0
#         dMin = fNorm2(x[i], xp[iMin])
#         for j in range(1, len(xp)):
#             d = fNorm2(x[i], xp[j])
#             if d < dMin:
#                 dMin = d
#                 iMin = j
#         if iMin == i:
#             match = match + 1
#     return (match, len(x))

# distance based record linkage
def sdcRecordLinkage(df_anon: pd.DataFrame, df_orig: pd.DataFrame, QIs: list):
    x = df_anon[QIs].to_numpy()
    xp = df_orig[QIs].to_numpy()

    match = 0

    for i_x, r1 in enumerate(x):
        # Given a record r1 in x, we compute its distance to each record r2 in xp
        sum_squares = np.sum((r1 - xp) ** 2, axis=1)

        # Then, we select the most similar to r1
        i_xp = np.argmin(sum_squares)

        if (i_x == i_xp):
            match += 1

    return (match, len(x))


def mondrian(df: pd.DataFrame, QIs: list, k: int) -> pd.DataFrame:
    start = timeit.default_timer()


    new_df = df.copy()
    db = new_df[QIs].to_numpy()
    assignedCl = mondrianCl(db, k)
    values = set(assignedCl)
    centroids = []

    max_value = db.max(axis=0)
    min_value = db.min(axis=0)
    ncp = 0

    for cl in values:
        rowsCl = selectFv(db, assignedCl, lambda x: x == cl)
        mat = np.array(rowsCl)

        max_row = mat.max(axis=0)
        min_row = mat.min(axis=0)

        # print(mat, mat.shape)
        # print(max_row)
        # print(min_row)


        ncp += sum(((max_row - min_row) * mat.shape[0]) / (max_value - min_value))

        meanRow = matrixColumnMeans(mat)
        centroids.append(meanRow)

    newDb = [centroids[assignedCl[i]] for i in range(0, len(db))]
    new_df[QIs] = newDb

    n_registers = new_df.shape[0]
    n_attributes = new_df.shape[1]
    ncp = ncp / (n_registers * n_attributes)
    print(f'ncp={ncp}')


    stop = timeit.default_timer()
    execution_time = stop - start
    print(f"Execution Time: {execution_time}")

    return new_df


def mondrianCl(db, k):
    clId, assignedCl = mondrianWithUnassigned(db, k, [-1] * len(db), 0, -1)
    return assignedCl


def mondrianWithUnassigned(db, k, assignedCl=None, firstNCl=0, stillToProcess=-1):
    if assignedCl == None:
        assignedCl = [-1] * len(db)
    # print(assignedCl)
    unassigned = selectFv(db, assignedCl, lambda x: x == stillToProcess)
    if len(unassigned) < 2 * k:
        index2Update = list(
            filter(lambda i: assignedCl[i] == stillToProcess, range(0, len(assignedCl)))
        )
        newClNumber = firstNCl
        newAssignedCl = updateCl(index2Update, newClNumber, assignedCl)
        return (firstNCl + 1, newAssignedCl)
    else:
        loMaxMinusMin = matrixColumnFunction(unassigned, lambda x: (max(x) - min(x)))
        selectedAttribute = loMaxMinusMin.index(max(loMaxMinusMin))
        cutPoint = partitionAttributeValue(unassigned, selectedAttribute)
        iLow, iHigh = indexLowMidHighValues(
            db, assignedCl, stillToProcess, cutPoint, selectedAttribute
        )
        clLowValues = stillToProcess * 2
        newAssignedCl = updateCl(iLow, clLowValues, assignedCl)
        new1FirstNCl, new1AssignedCl = mondrianWithUnassigned(
            db, k, newAssignedCl, firstNCl, clLowValues
        )
        clHighValues = stillToProcess * 2 + 1
        allAssignedCl = updateCl(iHigh, clHighValues, newAssignedCl)
        new2FirstNCl, new2AssignedCl = mondrianWithUnassigned(
            db, k, new1AssignedCl, new1FirstNCl, clHighValues
        )
        return (new2FirstNCl, new2AssignedCl)



def partitionAttributeValue(unassigned, selectedAttribute):
    allValues = list(map(lambda x: x[selectedAttribute], unassigned))
    allValues.sort()
    midValue = allValues[len(allValues) // 2]
    # midValue = (allValues[0]+allValues[len(allValues)-1])/2
    return midValue


def indexLowMidHighValues(db, assignedCl, idCl, midValue, selectedAttribute):
    indexLowValues = list(
        filter(
            lambda i: assignedCl[i] == idCl and db[i][selectedAttribute] < midValue,
            range(0, len(db)),
        )
    )
    indexMidValues = list(
        filter(
            lambda i: assignedCl[i] == idCl and db[i][selectedAttribute] == midValue,
            range(0, len(db)),
        )
    )
    indexHighValues = list(
        filter(
            lambda i: assignedCl[i] == idCl and db[i][selectedAttribute] > midValue,
            range(0, len(db)),
        )
    )
    numberValues = len(indexLowValues) + len(indexMidValues) + len(indexHighValues)
    toLow = max(numberValues // 2 - len(indexLowValues), 0)
    # print(midValue)
    # print(toLow)
    indexLow = indexLowValues + indexMidValues[0:toLow]
    indexHigh = indexMidValues[toLow:] + indexHighValues
    return (indexLow, indexHigh)


def main():
    df = pd.read_csv("./datasets/wearable-exercise-frailty.csv")
    QIs = ["Age", "Height(cm)", "Weight(kg)"]
    print(df)
    print(mdav(df, QIs, 9))

if __name__ == "__main__":
    main()
