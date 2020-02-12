import numpy as np
import torch
import sys
import json

from FiniteElementMesh import FiniteElementMesh
from ProjectiveDynamicsSolver import ProjectiveDynamicsSolver

class LatticeMesh(FiniteElementMesh):
    def __init__(self, simProperties):
        self.width = simProperties["width"]
        self.height = simProperties["height"]
        self.depth = simProperties["depth"]
        self.gridDx = simProperties["gridDx"]
        self.fileName = simProperties["targetFile"]
        self.activeCells = []
        #populate active cells
        for i in range(0, self.width):
            for j in range(0, self.height):
                for k in range(0, self.depth):
                    self.activeCells += [[i, j, k]]
        #get particles for activeCells
        self.activeParticles = {}
        self.particles = []
        self.particleIndex = {}
        #set left and right handles
        self.leftHandleIndices = []
        self.rightHandleIndices = []
        for cell in self.activeCells:
            for i in range(cell[0], cell[0]+2):
                for j in range(cell[1], cell[1]+2):
                    for k in range(cell[2], cell[2]+2):
                        pKey = tuple([i, j, k]) # dict(particle=[i, j, k], numParticle=len(self.particles))
                        if not self.findParticle(self.activeParticles, pKey):
                            index = len(self.particles)
                            self.activeParticles[pKey] = index
                            self.particles += [[i*self.gridDx, j*self.gridDx, k*self.gridDx]]
                            self.particleIndex[((i * (self.height + 1) * (self.depth + 1)) + (j * (self.depth + 1)) + k)] = index
                            if i == 0:
                                self.leftHandleIndices.append(index)
                            elif i == self.width:
                                self.rightHandleIndices.append(index)
        #initialise mesh elements
        #populate mesh elements
        self.meshElements = []
        for cell in self.activeCells:
            pCell = []
            for i in range(0, 2):
                for j in range(0, 2):
                    for k in range(0, 2):
                        pKey = tuple([cell[0]+i, cell[1]+j, cell[2]+k])
                        pNum = self.getParticle(self.activeParticles, pKey)
                        if pNum is None:
                            print("ERROR: PARTICLE NOT FOUND")
                            sys.exit()
                        else:
                            pCell.append(pNum)
            self.meshElements.append([pCell[0], pCell[4], pCell[6], pCell[7]])
            self.meshElements.append([pCell[0], pCell[4], pCell[7], pCell[5]])
            self.meshElements.append([pCell[0], pCell[5], pCell[7], pCell[1]])
            self.meshElements.append([pCell[0], pCell[7], pCell[3], pCell[1]])
            self.meshElements.append([pCell[0], pCell[7], pCell[2], pCell[3]])
            self.meshElements.append([pCell[0], pCell[6], pCell[2], pCell[7]])

        #set left handle and right handle velocity
        self.leftHandleVelocity = np.zeros(3)
        for v in simProperties["leftHandleVelocity"]:
            self.leftHandleVelocity[0] = v["x"]
            self.leftHandleVelocity[1] = v["y"]
            self.leftHandleVelocity[2] = v["z"]
        self.rightHandleVelocity = np.zeros(3)
        for v in simProperties["rightHandleVelocity"]:
            self.rightHandleVelocity[0] = v["x"]
            self.rightHandleVelocity[1] = v["y"]
            self.rightHandleVelocity[2] = v["z"]

    def writeToFile(self, i, pos):
        if i == 0:
            f = open(self.fileName, "w")
        else:
            f = open(self.fileName, "a")
        np.savetxt(f, pos, delimiter=',')
        f.close()

    def findParticle(self, pDict, pKey):
        try:
            ans = pDict[pKey]
            return True
        except:
            return False

    def getParticle(self, pDict, pKey):
        try:
            return pDict[pKey]
        except:
            return None

    def resetConstrainedParticles(self, x, value):
        for index in self.leftHandleIndices:
            x[index] = value
        for index in self.rightHandleIndices:
            x[index] = value
        return x

    def setBoundaryConditions(self, pos, vel, stepEndTime):
        effectiveTime = min(stepEndTime, 1.0)
        for p in self.leftHandleIndices:
            pos[p] = self.particles[p] + effectiveTime * self.leftHandleVelocity
            vel[p] = self.leftHandleVelocity
        for p in self.rightHandleIndices:
            pos[p] = self.particles[p] + effectiveTime * self.rightHandleVelocity
            vel[p] = self.rightHandleVelocity

    def sortParticles(self):
        yx = zip(self.particleIndex, self.particles)
        yx = sorted(yx)
        self.sortedParticles = [x for y,x in yx]
        print(self.sortedParticles)
        self.sortedIndex = [y for y,x in yx]

if __name__ == '__main__':
    simProperties = None
    with open('properties.json') as json_file:
        simProperties = json.load(json_file)
    lm = LatticeMesh(simProperties)
    fem = FiniteElementMesh(density=1.e2, mu=1., lmbda=4., rayleighCoefficient=.05, frames=50, frameDt=0.1)
    #lm.sortParticles()
    fem.setParticles(lm.particles, lm.width, lm.height, lm.depth, lm.particleIndex)
    fem.setMeshElements(lm.meshElements)
    fem.setHandles(lm.leftHandleIndices, lm.rightHandleIndices)
    fem.registerLatticeMeshObject(lm)
    fem.initialiseUndeformedConfiguration()

    #ProjectiveDynamicsSolver
    print("pd")
    pdSolver = ProjectiveDynamicsSolver(simProperties, lm.particles, lm.particleIndex, lm.meshElements, lm)
    sys.exit()
    #testcase
    '''
    x = np.random.rand(len(lm.particles), 3)
    y1 = np.zeros(shape=[len(lm.particles), 3], dtype=np.float32)
    y1 = fem.multiplyWithStiffnessMatrixPD(x, y1)
    y2 = np.zeros(shape=[len(lm.particles), 3], dtype=np.float32)
    y2 = fem.multiplyWithLHSMatrix(x, y2)
    for i in range(0, len(lm.particles)):
        if y1[i][0] != y2[i][0]:
            print(str(y1[i][0]) + " " + str(y2[i][0]))
    '''
    for i in range(1, 50):
        fem.simulateFrame(i)
