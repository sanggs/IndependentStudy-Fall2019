import numpy as np
import torch
import sys
import json

# from FiniteElementMesh import FiniteElementMesh
from ProjectiveDynamicsSolver import ProjectiveDynamicsSolver

class LatticeMesh:
    def __init__(self, simProperties):
        self.width = simProperties["width"]
        self.height = simProperties["height"]
        self.depth = simProperties["depth"]
        self.gridDx = simProperties["gridDx"]
        self.fileName = simProperties["targetFile"]
        self.dimension = simProperties["dimension"]
        self.activeCells = []
        #populate active cells
        for i in range(0, self.width):
            for j in range(0, self.height):
                for k in range(0, self.depth):
                    self.activeCells += [[i, j, k]]
        #get particles for activeCells
        self.activeParticles = {}
        self.particles = torch.zeros([self.dimension, self.width+1, self.height+1, self.depth+1], dtype=torch.float32)
        self.particleIndex = {}
        #set left and right handles
        self.leftHandleIndices = []
        self.originalLeftHandlePositions = []
        self.rightHandleIndices = []
        self.originalRightHandlePositions = []
        count = 0
        for cell in self.activeCells:
            for i in range(cell[0], cell[0]+2):
                for j in range(cell[1], cell[1]+2):
                    for k in range(cell[2], cell[2]+2):
                        pKey = tuple([i, j, k]) # dict(particle=[i, j, k], numParticle=len(self.particles))
                        if not self.findParticle(self.activeParticles, pKey):
                            index = count
                            self.activeParticles[pKey] = index
                            self.particles[:, i, j, k] = torch.tensor([i*self.gridDx, j*self.gridDx, k*self.gridDx])
                            self.particleIndex[index] = pKey
                            if i == 0:
                                self.leftHandleIndices.append([i, j, k])
                                self.originalLeftHandlePositions.append([i*self.gridDx, j*self.gridDx, k*self.gridDx])
                            elif i == self.width:
                                self.rightHandleIndices.append([i, j, k])
                                self.originalRightHandlePositions.append([i*self.gridDx, j*self.gridDx, k*self.gridDx])
                            count += 1
        #initialise mesh elements
        #populate mesh elements
        self.meshElements = []
        for cell in self.activeCells:
            pCell = []
            for i in range(0, 2):
                for j in range(0, 2):
                    for k in range(0, 2):
                        pIndex = [cell[0]+i, cell[1]+j, cell[2]+k]
                        # pNum = self.getParticle(self.activeParticles, pKey)
                        # if pNum is None:
                        #     print("ERROR: PARTICLE NOT FOUND")
                        #     sys.exit()
                        # else:
                        #     pCell.append(pNum)
                        pCell.append(pIndex)
            self.meshElements.append([pCell[0], pCell[4], pCell[6], pCell[7]])
            self.meshElements.append([pCell[0], pCell[4], pCell[7], pCell[5]])
            self.meshElements.append([pCell[0], pCell[5], pCell[7], pCell[1]])
            self.meshElements.append([pCell[0], pCell[7], pCell[3], pCell[1]])
            self.meshElements.append([pCell[0], pCell[7], pCell[2], pCell[3]])
            self.meshElements.append([pCell[0], pCell[6], pCell[2], pCell[7]])
        self.meshElements = torch.tensor(self.meshElements, dtype=torch.int64)
        # for element in self.meshElements:
        #     print(element)
        #     print(self.particles[element[0][0], element[0][1], element[0][2]])
        #     print(self.particles[element[1][0], element[1][1], element[1][2]])
        #     print(self.particles[element[2][0], element[2][1], element[2][2]])
        #     print(self.particles[element[3][0], element[3][1], element[3][2]])

        #set left handle and right handle velocity
        self.leftHandleVelocity = torch.zeros(3)
        for v in simProperties["leftHandleVelocity"]:
            self.leftHandleVelocity[0] = v["x"]
            self.leftHandleVelocity[1] = v["y"]
            self.leftHandleVelocity[2] = v["z"]
        self.rightHandleVelocity = torch.zeros(3)
        for v in simProperties["rightHandleVelocity"]:
            self.rightHandleVelocity[0] = v["x"]
            self.rightHandleVelocity[1] = v["y"]
            self.rightHandleVelocity[2] = v["z"]

    def writeToFile(self, fno, pos):
        p = np.ones(shape = [pos.shape[1] * pos.shape[2] * pos.shape[3], pos.shape[0]], dtype=np.float32)
        for j in self.particleIndex:
            index = self.particleIndex[j]
            p[j, :] = pos[:, index[0], index[1], index[2]].numpy()
        if fno == 0:
            f = open(self.fileName, "w")
        else:
            f = open(self.fileName, "a")
        np.savetxt(f, p, delimiter=',')
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
            x[:, index[0], index[1], index[2]] = value
        for index in self.rightHandleIndices:
            x[:, index[0], index[1], index[2]] = value
        return x

    def setBoundaryConditions(self, pos, vel, stepEndTime):
        effectiveTime = min(stepEndTime, 1.0)
        print(effectiveTime)
        count = 0
        for p in self.leftHandleIndices:
            pos[:, p[0], p[1], p[2]] = torch.tensor(self.originalLeftHandlePositions[count]) + effectiveTime * self.leftHandleVelocity
            vel[:, p[0], p[1], p[2]] = self.leftHandleVelocity
            count += 1
        count = 0
        for p in self.rightHandleIndices:
            pos[:, p[0], p[1], p[2]] = torch.tensor(self.originalRightHandlePositions[count]) + effectiveTime * self.rightHandleVelocity
            vel[:, p[0], p[1], p[2]] = self.rightHandleVelocity
            count += 1

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
    # fem = FiniteElementMesh(density=1.e2, mu=1., lmbda=4., rayleighCoefficient=.05, frames=50, frameDt=0.1)
    #lm.sortParticles()
    # fem.setParticles(lm.particles, lm.width, lm.height, lm.depth, lm.particleIndex)
    # fem.setMeshElements(lm.meshElements)
    # fem.setHandles(lm.leftHandleIndices, lm.rightHandleIndices)
    # fem.registerLatticeMeshObject(lm)
    # fem.initialiseUndeformedConfiguration()

    #ProjectiveDynamicsSolver
    print("pd")
    pdSolver = ProjectiveDynamicsSolver(simProperties, lm.particles, lm.particleIndex, lm.meshElements, lm)

    #testcase
    
    x = torch.rand(3, lm.width+1, lm.height+1, lm.depth+1)
    y1 = torch.zeros(3, lm.width+1, lm.height+1, lm.depth+1)
    y1 = pdSolver.multiplyWithStiffnessMatrixPD(x, y1)
    y2 = torch.zeros(3, lm.width+1, lm.height+1, lm.depth+1)
    y2 = pdSolver.multiplyWithStencil(x, y2)
    error = torch.abs(y2.sub(y1))
    print(torch.sum(torch.where(error > 1e-6, torch.tensor(1), torch.tensor(0))))
    
    for i in range(1, 50):
        pdSolver.simulateFrame(i)
