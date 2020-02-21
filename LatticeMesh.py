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
        # set left and right handles --> not used in the torch implementation ,since we know what left and right handles are
        self.leftHandleIndices = []
        self.rightHandleIndices = []
        # set original positions of left and right handles
        self.originalHandlePositions = torch.zeros([self.dimension, 2, self.height+1, self.depth+1], dtype = torch.float32) #left handle is index 0, right handle is index 1
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
                                # self.originalLeftHandlePositions.append([i*self.gridDx, j*self.gridDx, k*self.gridDx])
                                self.originalHandlePositions[:, 0, j, k] = torch.tensor([i*self.gridDx, j*self.gridDx, k*self.gridDx])
                            elif i == self.width:
                                self.rightHandleIndices.append([i, j, k])
                                # self.originalRightHandlePositions.append([i*self.gridDx, j*self.gridDx, k*self.gridDx])
                                self.originalHandlePositions[:, 1, j, k] = torch.tensor([i*self.gridDx, j*self.gridDx, k*self.gridDx])
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
                        pCell.append(pIndex)
            self.meshElements.append([pCell[0], pCell[4], pCell[6], pCell[7]])
            self.meshElements.append([pCell[0], pCell[4], pCell[7], pCell[5]])
            self.meshElements.append([pCell[0], pCell[5], pCell[7], pCell[1]])
            self.meshElements.append([pCell[0], pCell[7], pCell[3], pCell[1]])
            self.meshElements.append([pCell[0], pCell[7], pCell[2], pCell[3]])
            self.meshElements.append([pCell[0], pCell[6], pCell[2], pCell[7]])
        self.meshElements = torch.tensor(self.meshElements, dtype=torch.int64)

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
        x[:, 0, :, :] = value
        x[:, self.width, :, :] = value

    def setBoundaryConditions(self, pos, vel, stepEndTime):
        effectiveTime = min(stepEndTime, 1.0)
        # LEFT HANDLE
        pos[0, 0, :, :] = self.originalHandlePositions[0, 0, :, :] + effectiveTime * self.leftHandleVelocity[0]
        pos[1, 0, :, :] = self.originalHandlePositions[1, 0, :, :] + effectiveTime * self.leftHandleVelocity[1]
        pos[2, 0, :, :] = self.originalHandlePositions[2, 0, :, :] + effectiveTime * self.leftHandleVelocity[2]
        # RIGHT HANDLE
        pos[0, self.width, :, :] = self.originalHandlePositions[0, 1, :, :] + effectiveTime * self.rightHandleVelocity[0]
        pos[1, self.width, :, :] = self.originalHandlePositions[1, 1, :, :] + effectiveTime * self.rightHandleVelocity[1]
        pos[2, self.width, :, :] = self.originalHandlePositions[2, 1, :, :] + effectiveTime * self.rightHandleVelocity[2]

    def sortParticles(self):
        yx = zip(self.particleIndex, self.particles)
        yx = sorted(yx)
        self.sortedParticles = [x for y,x in yx]
        print(self.sortedParticles)
        self.sortedIndex = [y for y,x in yx]

def testStencil():
    x = torch.rand(3, lm.width+1, lm.height+1, lm.depth+1)
    y1 = torch.zeros(3, lm.width+1, lm.height+1, lm.depth+1)
    y1 = pdSolver.multiplyWithStiffnessMatrixPD(x, y1)
    y2 = torch.zeros(3, lm.width+1, lm.height+1, lm.depth+1)
    y2 = pdSolver.multiplyWithStencil(x, y2)
    error = torch.abs(y2.sub(y1))
    print(torch.sum(torch.where(error > 1e-7, torch.tensor(1), torch.tensor(0))))

if __name__ == '__main__':
    simProperties = None
    with open('properties.json') as json_file:
        simProperties = json.load(json_file)
    lm = LatticeMesh(simProperties)

    #ProjectiveDynamicsSolver
    print("pd")
    pdSolver = ProjectiveDynamicsSolver(simProperties, lm.particles, lm.particleIndex, lm.meshElements, lm)

    #test the accuracy of the precomputed stencil
    # testStencil()
    
    for i in range(1, 50):
        pdSolver.simulateFrame(i)
