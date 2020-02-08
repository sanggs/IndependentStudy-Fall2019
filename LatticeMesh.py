import numpy as np
import torch
import sys

from FiniteElementMesh import FiniteElementMesh

class LatticeMesh(FiniteElementMesh):
    def __init__(self):
        self.width = 4
        self.height = 4
        self.depth = 4
        self.gridDx = 0.5
        self.activeCells = []
        #populate active cells
        for i in range(0, self.width):
            for j in range(0, self.height):
                for k in range(0, self.depth):
                    self.activeCells += [[i, j, k]]
        #get particles for activeCells
        self.activeParticles = []
        self.particles = []
        self.particleIndex = []
        for cell in self.activeCells:
            for i in range(cell[0], cell[0]+2):
                for j in range(cell[1], cell[1]+2):
                    for k in range(cell[2], cell[2]+2):
                        insertCell = dict(particle=[i, j, k], numParticle=len(self.particles))
                        if not self.findParticle(self.activeParticles, insertCell):
                            self.activeParticles.append(insertCell)
                            self.particles += [[i*self.gridDx, j*self.gridDx, k*self.gridDx]]
                            self.particleIndex += [(i * 25) + (j * 5) + k]
        #initialise mesh elements
        #populate mesh elements
        self.meshElements = []
        for cell in self.activeCells:
            pCell = []
            for i in range(0, 2):
                for j in range(0, 2):
                    for k in range(0, 2):
                        insertCell = [cell[0]+i, cell[1]+j, cell[2]+k]
                        pNum = self.getParticle(self.activeParticles, insertCell)
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

        #set left and right handles
        self.leftHandleIndices = []
        self.rightHandleIndices = []
        for entry in self.activeParticles:
            cell = entry['particle']
            if cell[0] == 0:
                self.leftHandleIndices.append(entry['numParticle'])
            elif cell[0] == self.width:
                self.rightHandleIndices.append(entry['numParticle'])
        #set left handle and right handle velocity
        self.leftHandleVelocity = np.array([-.5, 0.0, 0.0], dtype = np.float32)
        self.rightHandleVelocity = np.array([.5, 0.0, 0.0], dtype = np.float32)

    def writeToFile(self, i, pos):
        if i == 0:
            f = open("3DPoints.csv", "w")
        else:
            f = open("3DPoints.csv", "a")
        np.savetxt(f, pos, delimiter=',')
        f.close()

    def findParticle(self, pList, p):
        for item in pList:
            if item['particle'] == p['particle']:
                return True
        return False

    def getParticle(self, pList, p):
        for item in pList:
            if item['particle'] == p:
                return item['numParticle']
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
    lm = LatticeMesh()
    fem = FiniteElementMesh(density=1.e2, mu=1., lmbda=4., rayleighCoefficient=.05, frames=50, frameDt=0.1)
    #lm.sortParticles()
    fem.setParticles(lm.particles, lm.width, lm.height, lm.depth, lm.particleIndex)
    fem.setMeshElements(lm.meshElements)
    fem.setHandles(lm.leftHandleIndices, lm.rightHandleIndices)
    fem.registerLatticeMeshObject(lm)
    fem.initialiseUndeformedConfiguration()
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
