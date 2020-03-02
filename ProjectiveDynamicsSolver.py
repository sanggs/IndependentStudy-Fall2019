import torch
import numpy as np
import sys
import time

from ConjugateGradientSolver import CGSolver

class ProjectiveDynamicsSolver:
    def __init__(self, simProperties, particles, particleIndex, meshElements, interiorCellMeshElements, lm=None):
        self.density = float(simProperties["density"])
        self.mu = simProperties["mu"]
        self.lmbda = simProperties["lmbda"]
        self.rayleighCoefficient = simProperties["rayleighCoefficient"]
        self.nFrames = simProperties["frames"]
        self.subSteps = simProperties["subSteps"]
        self.frameDt = simProperties["frameDt"]
        self.width = simProperties["width"]
        self.height = simProperties["height"]
        self.depth = simProperties["depth"]
        self.dimension = simProperties["dimension"]
        self.particleMass = None
        self.particleVelocity = None
        self.dMInverse = None
        self.restVolume = None
        self.interiorCellMeshElements = None

        self.setParticles(particles, particleIndex)

        self.setMeshElements(meshElements)

        if(interiorCellMeshElements):
            self.setInteriorCellMeshElements(interiorCellMeshElements)

        self.latticeMeshObject = None
        if(lm):
            self.registerLatticeMeshObject(lm)

        self.timeMeasured = [] # timeMeasured is a list of tuples

        startTime = time.time() # start the timer
        self.initialiseUndeformedConfiguration()
        self.timeMeasured.append(tuple(['initialiseUndeformedConfiguration', time.time()-startTime])) # end the timer, add to the list


    def setParticles(self, particles, particleIndex):
        self.particles = particles
        self.numParticles = particles.shape[1]*particles.shape[2]*particles.shape[3] #shape0 = coord, shape1 = width, shape2 = height, shape3 = depth
        print(self.particles.shape)
        self.particleIndex = torch.utils.data.DataLoader(particleIndex)

    def setMeshElements(self, meshElements):
        self.meshElements = meshElements
        self.numMeshElements = self.meshElements.shape[0] #shape0 = number of elements, shape1 = dimensions+1, shape2 = coord

    def registerLatticeMeshObject(self, lm):
        self.latticeMeshObject = lm

    def setInteriorCellMeshElements(self, interiorCellMeshElements):
        self.interiorCellMeshElements = interiorCellMeshElements

    def getBuilderTensor(self):
        builder = torch.zeros(size=[self.dimension+1,self.dimension], dtype = torch.float32)
        builder[1:4, :] = torch.eye(3)
        builder[0, :] = -1 * torch.ones(3)
        return builder

    def initialiseUndeformedConfiguration(self):
        if self.particles is None or self.meshElements is None:
            print("Particles and meshElements not set")
            sys.exit()#sanity check for particles and meshElements
        # Builder Matrix: needed to compute dM from X
        builder = self.getBuilderTensor()
        #Velocity
        self.particleVelocity = torch.zeros([self.dimension, self.width+1, self.height+1, self.depth+1], dtype=torch.float32)
        #dM
        # dM = torch.zeros(size = [self.numMeshElements, 3, 3], dtype=torch.float32)
        # X = torch.zeros(size=[self.numMeshElements, 3, 4], dtype = torch.float32)
        self.dMInverse = torch.zeros([self.numMeshElements, 3, 3], dtype=torch.float32)
        #Volume
        self.restVolume = torch.zeros(size = [6, self.width, self.height, self.depth], dtype=torch.float32)

        # for i in range(0, self.numMeshElements):
        #     #we know meshElements will have 4 elements
        #     for j in range(0, self.dimension+1):
        #         X[i, :, j] = self.particles[:, self.meshElements[i][j][0], self.meshElements[i][j][1], self.meshElements[i][j][2]]

        #access mesh elements without loop
        X = torch.zeros(size = [self.width, self.height, self.depth, 6, self.dimension, self.dimension+1], dtype=torch.float32)
        for i in range(0, 3):
            #Tet 0: P0 P4 P6 P7 [000, 100, 110, 111]
            X[:, :, :, 0, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 0, i, 1] = self.particles[i, 2:self.width+2, 1:self.height+1, 1:self.depth+1] #P4 100
            X[:, :, :, 0, i, 2] = self.particles[i, 2:self.width+2, 2:self.height+2, 1:self.depth+1] #P6 110
            X[:, :, :, 0, i, 3] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            #Tet 1: P0 P4 P7 P5 [000, 100, 111, 101]
            X[:, :, :, 1, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 1, i, 1] = self.particles[i, 2:self.width+2, 1:self.height+1, 1:self.depth+1] #P4 100
            X[:, :, :, 1, i, 2] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 1, i, 3] = self.particles[i, 2:self.width+2, 1:self.height+1, 2:self.depth+2] #P5 101
            #Tet 2: P0 P5 P7 P1 [000, 101, 111, 001]
            X[:, :, :, 2, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 2, i, 1] = self.particles[i, 2:self.width+2, 1:self.height+1, 2:self.depth+2] #P5 101
            X[:, :, :, 2, i, 2] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 2, i, 3] = self.particles[i, 1:self.width+1, 1:self.height+1, 2:self.depth+2] #P1 001
            #Tet 3: P0 P7 P3 P1 [000, 111, 011, 001]
            X[:, :, :, 3, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 3, i, 1] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 3, i, 2] = self.particles[i, 1:self.width+1, 2:self.height+2, 2:self.depth+2] #P3 011
            X[:, :, :, 3, i, 3] = self.particles[i, 1:self.width+1, 1:self.height+1, 2:self.depth+2] #P1 001
            #Tet 4: P0 P7 P2 P3 [000, 111, 010, 011]
            X[:, :, :, 4, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 4, i, 1] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 4, i, 2] = self.particles[i, 1:self.width+1, 2:self.height+2, 1:self.depth+1] #P2 010
            X[:, :, :, 4, i, 3] = self.particles[i, 1:self.width+1, 2:self.height+2, 2:self.depth+2] #P3 011
            #Tet 5: P0 P6 P2 P7 [000, 110, 010, 111]
            X[:, :, :, 5, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 5, i, 1] = self.particles[i, 2:self.width+2, 2:self.height+2, 1:self.depth+1] #P6 110
            X[:, :, :, 5, i, 2] = self.particles[i, 1:self.width+1, 2:self.height+2, 1:self.depth+1] #P2 010
            X[:, :, :, 5, i, 3] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111

        dM = torch.matmul(X, builder)
        self.dMInverse = torch.inverse(dM)
        self.restVolume = 0.5 * torch.det(dM)
        #Compute Gtranspose
        self.GTranspose = torch.matmul(builder, self.dMInverse)
        #Mass => Not needed for PD
        # self.particleMass = torch.zeros(self.numParticles, dtype=torch.float32)
        #Compute particle mass => Not needed for PD
        # for i in range(0, self.numMeshElements):
        #     elementMass = self.density * self.restVolume[i]
        #     for particle in self.meshElements[i]:
        #         self.particleMass[particle] += (1.0/5.0) * elementMass

        #write points to file
        # startTime = time.time() # start the timer
        # self.latticeMeshObject.writeToFile(0, self.particles)
        # print(time.time()-startTime)
        # self.timeMeasured.append(tuple(['writeToFile', time.time()-startTime])) # end the timer, add to the list
        #Extract stencil
        startTime = time.time() # start the timer
        self.precomputeStencilMatrix()
        print(time.time()-startTime)
        self.timeMeasured.append(tuple(['preComputeStencilMatrix', time.time()-startTime])) # end the timer, add to the list

    def printStencil(self):
        for i in range(0, self.width+1):
            for j in range(0, self.height+1):
                for k in range(0, self.depth+1):
                    print(str(i) + " " + str(j) + " " + str(k))
                    print(self.stencil[i][j][k])
        sys.exit(0)
        return

    def precomputeStencilMatrix(self):
        #print(self.particleIndex.dataset)
        startTime = time.time()
        self.stencil = torch.zeros(self.width+1, self.height+1, self.depth+1, self.dimension, self.dimension, self.dimension)
        G = torch.transpose(self.GTranspose, 4, 5)
        GtG = torch.matmul(self.GTranspose, G)
        # print(time.time()-startTime)
        self.timeMeasured.append(tuple(['gTransposeGComputation', time.time()-startTime])) # end the timer, add to the list

        startTime = time.time()
        self.stencil1 = torch.zeros(2, 2, 2, self.dimension, self.dimension, self.dimension)
        cellIndex = torch.tensor(self.latticeMeshObject.interiorActiveCell, dtype=torch.int64)
        print(cellIndex)
        meshElements = torch.tensor(self.latticeMeshObject.interiorCellMeshElements, dtype=torch.int64)
        for i in range(0, 6):
            # print(meshElements[i])
            w = 2 * self.mu * self.restVolume[cellIndex[0], cellIndex[1], cellIndex[2], i] * GtG[cellIndex[0], cellIndex[1], cellIndex[2], i]
            # print(i)
            # print(w)
            #diagonals
            for j in range(0, self.dimension+1):
                index = meshElements[i][j] - cellIndex
                # print(index)
                self.stencil1[index[0], index[1], index[2], 1, 1, 1] += w[j][j]
                # print(self.stencil1[index[0]-1, index[1]-1, index[2]-1, 1, 1, 1])
            #upper
            for row in range(0, self.dimension+1):
                for col in range(row+1, self.dimension+1):
                    d = meshElements[i][col]-meshElements[i][row] + 1
                    index = meshElements[i][row] - cellIndex
                    self.stencil1[index[0], index[1], index[2], d[0], d[1], d[2]] += w[row][col]
                    # print(self.stencil1[index[0]-1, index[1]-1, index[2]-1, d[0], d[1], d[2]])
                    d = meshElements[i][row]-meshElements[i][col] + 1
                    index = meshElements[i][col] - cellIndex
                    self.stencil1[index[0], index[1], index[2], d[0], d[1], d[2]] += w[col][row]
                    # print(self.stencil1[index[0]-1, index[1]-1, index[2]-1, d[0], d[1], d[2]])
        # print(time.time()-startTime)
        self.timeMeasured.append(tuple(['computingWeightsForOneCell', time.time()-startTime])) # end the timer, add to the list
        startTime = time.time()
        #Repeat and add
        x_start = 0
        y_start = 0
        z_start = 0
        x_end = self.width
        y_end = self.height
        z_end = self.depth
        for i in range(0, 2):
            for j in range(0, 2):
                for k in range(0, 2):
                    self.stencil[x_start+i:x_end+i, y_start+j:y_end+j, z_start+k:z_end+k, :, :, :] += self.stencil1[i,j,k,:,:,:]
        # print(time.time()-startTime)
        # self.printStencil()
        self.timeMeasured.append(tuple(['populatingTheStencil', time.time()-startTime])) # end the timer, add to the list
        return

    def precomputeStencilWithoutInteriorCell(self):
        #print(self.particleIndex.dataset)
        self.stencil2 = torch.zeros(self.width+1, self.height+1, self.depth+1, self.dimension, self.dimension, self.dimension)
        startTime = time.time()
        G = torch.transpose(self.GTranspose, )
        GtG = torch.matmul(self.GTranspose, G)
        self.timeMeasured.append(tuple(['gTransposeGComputation', time.time()-startTime])) # end the timer, add to the list
        startTime = time.time()
        for i in range(0, self.numMeshElements):
            w = 2 * self.mu * self.restVolume[i] * GtG[i]
            #diagonals
            for j in range(0, self.dimension+1):
                self.stencil2[self.meshElements[i][j][0]-1, self.meshElements[i][j][1]-1, self.meshElements[i][j][2]-1, 1, 1, 1] += w[j][j]
            #upper
            for row in range(0, self.dimension+1):
                for col in range(row+1, self.dimension+1):
                    d = self.meshElements[i][col]-self.meshElements[i][row] + 1
                    self.stencil2[self.meshElements[i][row][0]-1, self.meshElements[i][row][1]-1, self.meshElements[i][row][2]-1, d[0], d[1], d[2]] += w[row][col]
                    d = self.meshElements[i][row]-self.meshElements[i][col] + 1
                    self.stencil2[self.meshElements[i][col][0]-1, self.meshElements[i][col][1]-1, self.meshElements[i][col][2]-1, d[0], d[1], d[2]] += w[row][col]
            # for row in range(0, self.dimension+1):
            #     for col in range(0, self.dimension+1):
            #         d = self.meshElements[i, col] - self.meshElements[i, row] + 1
            #         self.stencil[self.meshElements[i, row, 0], self.meshElements[i, row, 1], self.meshElements[i, row, 2], d[0], d[1], d[2]] += w[row][col]
        self.timeMeasured.append(tuple(['populatingTheStencil', time.time()-startTime])) # end the timer, add to the list
        return

    def multiplyWithStiffnessMatrixPD(self, p, f):
        X = torch.zeros(size = [self.width, self.height, self.depth, 6, self.dimension, self.dimension+1], dtype=torch.float32)
        for i in range(0, 3):
            #Tet 0: P0 P4 P6 P7 [000, 100, 110, 111]
            X[:, :, :, 0, i, 0] = p[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 0, i, 1] = p[i, 2:self.width+2, 1:self.height+1, 1:self.depth+1] #P4 100
            X[:, :, :, 0, i, 2] = p[i, 2:self.width+2, 2:self.height+2, 1:self.depth+1] #P6 110
            X[:, :, :, 0, i, 3] = p[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            #Tet 1: P0 P4 P7 P5 [000, 100, 111, 101]
            X[:, :, :, 1, i, 0] = p[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 1, i, 1] = p[i, 2:self.width+2, 1:self.height+1, 1:self.depth+1] #P4 100
            X[:, :, :, 1, i, 2] = p[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 1, i, 3] = p[i, 2:self.width+2, 1:self.height+1, 2:self.depth+2] #P5 101
            #Tet 2: P0 P5 P7 P1 [000, 101, 111, 001]
            X[:, :, :, 2, i, 0] = p[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 2, i, 1] = p[i, 2:self.width+2, 1:self.height+1, 2:self.depth+2] #P5 101
            X[:, :, :, 2, i, 2] = p[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 2, i, 3] = p[i, 1:self.width+1, 1:self.height+1, 2:self.depth+2] #P1 001
            #Tet 3: P0 P7 P3 P1 [000, 111, 011, 001]
            X[:, :, :, 3, i, 0] = p[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 3, i, 1] = p[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 3, i, 2] = p[i, 1:self.width+1, 2:self.height+2, 2:self.depth+2] #P3 011
            X[:, :, :, 3, i, 3] = p[i, 1:self.width+1, 1:self.height+1, 2:self.depth+2] #P1 001
            #Tet 4: P0 P7 P2 P3 [000, 111, 010, 011]
            X[:, :, :, 4, i, 0] = p[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 4, i, 1] = p[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 4, i, 2] = p[i, 1:self.width+1, 2:self.height+2, 1:self.depth+1] #P2 010
            X[:, :, :, 4, i, 3] = p[i, 1:self.width+1, 2:self.height+2, 2:self.depth+2] #P3 011
            #Tet 5: P0 P6 P2 P7 [000, 110, 010, 111]
            X[:, :, :, 5, i, 0] = p[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 5, i, 1] = p[i, 2:self.width+2, 2:self.height+2, 1:self.depth+1] #P6 110
            X[:, :, :, 5, i, 2] = p[i, 1:self.width+1, 2:self.height+2, 1:self.depth+1] #P2 010
            X[:, :, :, 5, i, 3] = p[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
        deformationF = torch.matmul(X, self.GTranspose)
        P = 2 * self.mu * deformationF
        G = torch.transpose(self.GTranspose, 4, 5)
        Q = torch.matmul(P, G)
        for i in range(0,3):
            for j in range(0,4):
                Q[:,:,:,:,i,j] = Q[:,:,:,:,i,j] * self.restVolume
        for i in range(0, 3):
            #Tet 0: P0 P4 P6 P7 [000, 100, 110, 111]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 0, i , 0] #P0 000
            f[i, 1:self.width+1, 0:self.height, 0:self.depth] += Q[:, :, :, 0, i , 1] #P4 100
            f[i, 1:self.width+1, 1:self.height+1, 0:self.depth] += Q[:, :, :, 0, i , 2] #P6 110
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 0, i , 3] #P7 111
            #Tet 1: P0 P4 P7 P5 [000, 100, 111, 101]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 1, i , 0] #P0 000
            f[i, 1:self.width+1, 0:self.height, 0:self.depth] += Q[:, :, :, 1, i , 1] #P4 100
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 1, i , 2] #P7 111
            f[i, 1:self.width+1, 0:self.height, 1:self.depth+1] += Q[:, :, :, 1, i , 3] #P5 101
            #Tet 2: P0 P5 P7 P1 [000, 101, 111, 001]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 2, i , 0] #P0 000
            f[i, 1:self.width+1, 0:self.height, 1:self.depth+1] += Q[:, :, :, 2, i , 1] #P5 101
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 2, i , 2] #P7 111
            f[i, 0:self.width, 0:self.height, 1:self.depth+1] += Q[:, :, :, 2, i , 3] #P1 001
            #Tet 3: P0 P7 P3 P1 [000, 111, 011, 001]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 3, i , 0] #P0 000
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 3, i , 1] #P7 111
            f[i, 0:self.width, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 3, i , 2] #P3 011
            f[i, 0:self.width, 0:self.height, 1:self.depth+1] += Q[:, :, :, 3, i , 3] #P1 001
            #Tet 4: P0 P7 P2 P3 [000, 111, 010, 011]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 4, i , 0] #P0 000
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 4, i , 1] #P7 111
            f[i, 0:self.width, 1:self.height+1, 0:self.depth] += Q[:, :, :, 4, i , 2] #P2 010
            f[i, 0:self.width, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 4, i , 3] #P3 011
            #Tet 5: P0 P6 P2 P7 [000, 110, 010, 111]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 5, i , 0] #P0 000
            f[i, 1:self.width+1, 1:self.height+1, 0:self.depth] += Q[:, :, :, 5, i , 1] #P6 110
            f[i, 0:self.width, 1:self.height+1, 0:self.depth] += Q[:, :, :, 5, i , 2] #P2 010
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 5, i , 3] #P7 111
        # for i in range(0, self.numMeshElements):
        #     X = torch.zeros(size=[3, 4], dtype = torch.float32)
        #     for j in range(0, self.dimension+1):
        #         X[:, j] = p[:, self.meshElements[i][j][0], self.meshElements[i][j][1], self.meshElements[i][j][2]]
        #     deformationF = torch.mm(X, self.GTranspose[i])
        #     P = 2 * self.mu * deformationF
        #     Q = self.restVolume[i] * torch.mm(P, self.GTranspose[i].t())
        #     for j in range(0, 4):
                # f[:, self.meshElements[i][j][0]-1, self.meshElements[i][j][1]-1, self.meshElements[i][j][2]-1] += Q[:, j]
        return

    def multiplyWithStencil(self, x, q):
        startTime = time.time() # start the timer

        q[:, :, :, :] = 0.0

        x_start = 0
        y_start = 0
        z_start = 0
        x_end = self.width+1
        y_end = self.height+1
        z_end = self.depth+1
        for c in range(0, 3):
            for di in range(0, 3):
                for dj in range(0, 3):
                    for dk in range(0, 3):
                        q[c, :, :, :] += x[c, x_start+di:x_end+di, y_start+dj:y_end+dj, z_start+dk:z_end+dk] * self.stencil[:, :, :, di, dj, dk]

        self.timeMeasured.append(tuple(['multiplyWithStencil', time.time()-startTime])) # end the timer, add to the list
        return

    #not used
    def computeDs(self, elementNum):
        builder = self.getBuilderTensor()
        X = torch.zeros(size=[3, 4], dtype = torch.float32)
        for j in range(0, self.dimension+1):
            X[:, j] = self.particles[:, self.meshElements[elementNum][j][0], self.meshElements[elementNum][j][1], self.meshElements[elementNum][j][2]]
        return torch.matmul(X, builder)

    def solveLocalStep(self):
        self.R = torch.zeros([self.numMeshElements, 3, 3], dtype = torch.float32)
        # Builder Matrix: needed to compute dM from X
        for i in range(0, self.numMeshElements):
            # dS = self.computeDs(i)
            X = torch.zeros(size=[3, 4], dtype = torch.float32)
            for j in range(0, self.dimension+1):
                X[ :, j] = self.particles[:, self.meshElements[i][j][0], self.meshElements[i][j][1], self.meshElements[i][j][2]]
            deformationF = torch.matmul(X, self.GTranspose[i])
            u, sigma, v = torch.svd(deformationF)
            if torch.det(u) < 0:
                if torch.det(v) < 0:
                    v[:, 2] = -1 * v[:, 2]
                u[:, 2] = -1 * u[:, 2]
            elif torch.det(v) < 0:
                v[:, 2] = -1 * v[:, 2]
            self.R[i] = torch.matmul(u, v.t())

    def computeElasticForce(self, f):
        #access mesh elements without loop
        X = torch.zeros(size = [self.width, self.height, self.depth, 6, self.dimension, self.dimension+1], dtype=torch.float32)
        for i in range(0, 3):
            #Tet 0: P0 P4 P6 P7 [000, 100, 110, 111]
            X[:, :, :, 0, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 0, i, 1] = self.particles[i, 2:self.width+2, 1:self.height+1, 1:self.depth+1] #P4 100
            X[:, :, :, 0, i, 2] = self.particles[i, 2:self.width+2, 2:self.height+2, 1:self.depth+1] #P6 110
            X[:, :, :, 0, i, 3] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            #Tet 1: P0 P4 P7 P5 [000, 100, 111, 101]
            X[:, :, :, 1, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 1, i, 1] = self.particles[i, 2:self.width+2, 1:self.height+1, 1:self.depth+1] #P4 100
            X[:, :, :, 1, i, 2] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 1, i, 3] = self.particles[i, 2:self.width+2, 1:self.height+1, 2:self.depth+2] #P5 101
            #Tet 2: P0 P5 P7 P1 [000, 101, 111, 001]
            X[:, :, :, 2, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 2, i, 1] = self.particles[i, 2:self.width+2, 1:self.height+1, 2:self.depth+2] #P5 101
            X[:, :, :, 2, i, 2] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 2, i, 3] = self.particles[i, 1:self.width+1, 1:self.height+1, 2:self.depth+2] #P1 001
            #Tet 3: P0 P7 P3 P1 [000, 111, 011, 001]
            X[:, :, :, 3, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 3, i, 1] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 3, i, 2] = self.particles[i, 1:self.width+1, 2:self.height+2, 2:self.depth+2] #P3 011
            X[:, :, :, 3, i, 3] = self.particles[i, 1:self.width+1, 1:self.height+1, 2:self.depth+2] #P1 001
            #Tet 4: P0 P7 P2 P3 [000, 111, 010, 011]
            X[:, :, :, 4, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 4, i, 1] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
            X[:, :, :, 4, i, 2] = self.particles[i, 1:self.width+1, 2:self.height+2, 1:self.depth+1] #P2 010
            X[:, :, :, 4, i, 3] = self.particles[i, 1:self.width+1, 2:self.height+2, 2:self.depth+2] #P3 011
            #Tet 5: P0 P6 P2 P7 [000, 110, 010, 111]
            X[:, :, :, 5, i, 0] = self.particles[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] #P0 000
            X[:, :, :, 5, i, 1] = self.particles[i, 2:self.width+2, 2:self.height+2, 1:self.depth+1] #P6 110
            X[:, :, :, 5, i, 2] = self.particles[i, 1:self.width+1, 2:self.height+2, 1:self.depth+1] #P2 010
            X[:, :, :, 5, i, 3] = self.particles[i, 2:self.width+2, 2:self.height+2, 2:self.depth+2] #P7 111
        deformationF = torch.matmul(X, self.GTranspose)

        z = torch.svd(deformationF)
        u = z.U
        s = z.S
        vt = torch.transpose(z.V, 4,5)
        u_det = torch.det(u)
        vt_det = torch.det(vt)
        # det of u & v less than 0
        u_mask = torch.where(u_det < 0, torch.tensor(-1.0), torch.tensor(1.0))
        v_mask = torch.where(vt_det < 0, torch.tensor(-1.0), torch.tensor(1.0))
        # uplusv = u_mask + v_mask
        uv_mask = torch.where((u_det < 0) & (vt_det < 0), torch.tensor(-1.0), torch.tensor(1.0))
        # only det of u less than 0
        us_mask = torch.where((u_det < 0) & (vt_det > 0), torch.tensor(-1.0), torch.tensor(1.0))
        # only det of v less than 0
        vs_mask = torch.where((u_det > 0) & (vt_det < 0), torch.tensor(-1.0), torch.tensor(1.0))

        # print(torch.sum(us_mask * vs_mask))
        
        for i in range(0, 3):
            u[:, :, :, :, i, 2] = u[:, :, :, :, i, 2] * uv_mask
            vt[:, :, :, :, 2, i] = vt[:, :, :, :, 2, i] * uv_mask

        for i in range(0, 3):
            u[:, :, :, :, i, 2] = u[:, :, :, :, i, 2] * us_mask
        s[:, :, :, :, 2] = s[:, :, :, :, 2] * us_mask

        for i in range(0, 3):
            vt[:, :, :, :, 2, i] = vt[:, :, :, :, 2, i] * vs_mask
        s[:, :, :, :, 2] = s[:, :, :, :, 2] * vs_mask

        deformationF = torch.matmul(u, torch.matmul(torch.diag_embed(s), vt))
        R = torch.matmul(u, vt)

        P = 2 * self.mu * torch.sub(deformationF, R)
        G = torch.transpose(self.GTranspose, 4, 5)
        Q = torch.matmul(P, G)
        for i in range(0,3):
            for j in range(0,4):
                Q[:,:,:,:,i,j] = Q[:,:,:,:,i,j] * (-1.0 * self.restVolume)
        for i in range(0, 3):
            #Tet 0: P0 P4 P6 P7 [000, 100, 110, 111]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 0, i , 0] #P0 000
            f[i, 1:self.width+1, 0:self.height, 0:self.depth] += Q[:, :, :, 0, i , 1] #P4 100
            f[i, 1:self.width+1, 1:self.height+1, 0:self.depth] += Q[:, :, :, 0, i , 2] #P6 110
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 0, i , 3] #P7 111
            #Tet 1: P0 P4 P7 P5 [000, 100, 111, 101]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 1, i , 0] #P0 000
            f[i, 1:self.width+1, 0:self.height, 0:self.depth] += Q[:, :, :, 1, i , 1] #P4 100
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 1, i , 2] #P7 111
            f[i, 1:self.width+1, 0:self.height, 1:self.depth+1] += Q[:, :, :, 1, i , 3] #P5 101
            #Tet 2: P0 P5 P7 P1 [000, 101, 111, 001]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 2, i , 0] #P0 000
            f[i, 1:self.width+1, 0:self.height, 1:self.depth+1] += Q[:, :, :, 2, i , 1] #P5 101
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 2, i , 2] #P7 111
            f[i, 0:self.width, 0:self.height, 1:self.depth+1] += Q[:, :, :, 2, i , 3] #P1 001
            #Tet 3: P0 P7 P3 P1 [000, 111, 011, 001]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 3, i , 0] #P0 000
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 3, i , 1] #P7 111
            f[i, 0:self.width, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 3, i , 2] #P3 011
            f[i, 0:self.width, 0:self.height, 1:self.depth+1] += Q[:, :, :, 3, i , 3] #P1 001
            #Tet 4: P0 P7 P2 P3 [000, 111, 010, 011]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 4, i , 0] #P0 000
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 4, i , 1] #P7 111
            f[i, 0:self.width, 1:self.height+1, 0:self.depth] += Q[:, :, :, 4, i , 2] #P2 010
            f[i, 0:self.width, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 4, i , 3] #P3 011
            #Tet 5: P0 P6 P2 P7 [000, 110, 010, 111]
            f[i, 0:self.width, 0:self.height, 0:self.depth] += Q[:, :, :, 5, i , 0] #P0 000
            f[i, 1:self.width+1, 1:self.height+1, 0:self.depth] += Q[:, :, :, 5, i , 1] #P6 110
            f[i, 0:self.width, 1:self.height+1, 0:self.depth] += Q[:, :, :, 5, i , 2] #P2 010
            f[i, 1:self.width+1, 1:self.height+1, 1:self.depth+1] += Q[:, :, :, 5, i , 3] #P7 111
        # print("Done computing RHS")
        return

    def computeElasticForceWithLoop(self, forceTensor):
        for i in range(0, self.numMeshElements):
            X = torch.zeros(size=[3, 4], dtype = torch.float32)
            for j in range(0, self.dimension+1):
                X[:, j] = self.particles[:, self.meshElements[i][j][0], self.meshElements[i][j][1], self.meshElements[i][j][2]]
            deformationF = torch.mm(X, self.GTranspose[i])
            u, sigma, v = torch.svd(deformationF)
            if torch.det(u) < 0:
                u[:, 2] = -1 * u[:, 2]
                if torch.det(v) < 0:
                    v[:, 2] = -1 * v[:, 2]
                else:
                    sigma[2] = -1 * sigma[2]
                deformationF = torch.mm(u, torch.mm(torch.diag(sigma), v.t()))
            elif torch.det(v) < 0:
                v[:, 2] = -1 * v[:, 2]
                sigma[2] = -1 * sigma[2]
                deformationF = torch.mm(u, torch.mm(torch.diag(sigma), v.t()))
            R = torch.mm(u, v.t())
            P = 2 * self.mu * (deformationF - R)
            Q = -1 * self.restVolume[i] * torch.mm(P, self.GTranspose[i].t())
            for j in range(0, 4):
                forceTensor[:, self.meshElements[i][j][0]-1, self.meshElements[i][j][1]-1, self.meshElements[i][j][2]-1] += Q[:, j]
        return

    def resetConstrainedParticles(self, t, val):
        self.latticeMeshObject.resetConstrainedParticles(t, val)
        return

    def solveLocalAndGlobalStep(self):
        rhs = torch.zeros(size=[self.dimension, self.width+1, self.height+1, self.depth+1], dtype=torch.float32)
        dx = torch.zeros(size= [self.dimension, self.width+3, self.height+3, self.depth+3], dtype=torch.float32)
        q = torch.zeros(size = [self.dimension, self.width+1, self.height+1, self.depth+1], dtype=torch.float32)
        s = torch.zeros(size = [self.dimension, self.width+3, self.height+3, self.depth+3], dtype=torch.float32)
        r = torch.zeros(size = [self.dimension, self.width+1, self.height+1, self.depth+1], dtype=torch.float32)

        startTime = time.time()
        self.computeElasticForce(rhs)
        self.timeMeasured.append(tuple(['computeElasticForce', time.time()-startTime])) # end the timer, add to the list
        startTime = time.time()
        self.resetConstrainedParticles(rhs, 0.0)
        self.timeMeasured.append(tuple(['resetConstrainedParticles', time.time()-startTime])) # end the timer, add to the list
        #print("printing rhs")
        #print(rhs)

        startTime = time.time() # start the timer
        cg = CGSolver(particles=dx, rhs=rhs, q=q, s=s, r=r, numIteration=250, minConvergenceNorm=1e-5, width=self.width, height=self.height, depth=self.depth, femObject=self)
        #solve
        cg.solve()
        self.timeMeasured.append(tuple(['cgSolve', time.time()-startTime])) # end the timer, add to the list
        dx = cg.getSolution()

        #update position vector with result
        # print(dx.shape)
        self.particles[:, 1:self.width+2, 1:self.height+2, 1:self.depth+2] += dx[:, 1:self.width+2, 1:self.height+2, 1:self.depth+2]

        return

    def pdSimulation(self):
        self.latticeMeshObject.setBoundaryConditions(self.particles,
            self.particleVelocity, self.stepEndTime)
        for i in range(0, 1):
            startTime = time.time() # start the timer
            # self.solveLocalStep()
            self.solveLocalAndGlobalStep()
            self.timeMeasured.append(tuple(['solveLocalStep', time.time()-startTime])) # end the timer, add to the list
        return

    def simulateFrame(self, frameNumber):
        print("frameNumber: "+ str(frameNumber))
        self.stepDt = self.frameDt / float(self.subSteps)
        for i in range(1, self.subSteps+1):
            self.stepEndTime = self.frameDt * (frameNumber-1) + self.stepDt * i
            if frameNumber == 5:
                r = (-2 * torch.rand(self.dimension, self.width+1, self.height+1, self.depth+1)) + 1.0
                self.particles[:, 1:self.width+2, 1:self.height+2, 1:self.depth+2] += r
                # self.latticeMeshObject.writeToFile(i, self.particles)
            startTime = time.time() # start the timer
            self.pdSimulation()
            self.timeMeasured.append(tuple(['pdSimulation', time.time()-startTime])) # end the timer, add to the list
            # self.latticeMeshObject.writeToFile(i, self.particles)

    def getTimeMeasured(self):
        return self.timeMeasured
