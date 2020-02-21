import torch
import numpy as np
import sys
import time

from ConjugateGradientSolver import CGSolver

class ProjectiveDynamicsSolver:
    def __init__(self, simProperties, particles, particleIndex, meshElements, lm=None):
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
        self.setParticles(particles, particleIndex)
        self.setMeshElements(meshElements)
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

    def getBuilderTensor(self):
        builder = torch.zeros(size=[4,3], dtype = torch.float32)
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
        dM = torch.zeros(size = [self.numMeshElements, 3, 3], dtype=torch.float32)
        X = torch.zeros(size=[self.numMeshElements, 3, 4], dtype = torch.float32)
        self.dMInverse = torch.zeros([self.numMeshElements, 3, 3], dtype=torch.float32)
        #Volume
        self.restVolume = torch.zeros(self.numMeshElements, dtype=torch.float32)
        #Mass
        self.particleMass = torch.zeros(self.numParticles, dtype=torch.float32)
        for i in range(0, self.numMeshElements):
            #we know meshElements will have 4 elements
            for j in range(0, self.dimension+1):
                X[i, :, j] = self.particles[:, self.meshElements[i][j][0], self.meshElements[i][j][1], self.meshElements[i][j][2]]
        dM = torch.matmul(X, builder)
        self.dMInverse = torch.inverse(dM)
        self.restVolume = 0.5 * torch.det(dM)
        #Compute Gtranspose
        self.GTranspose = torch.matmul(builder, self.dMInverse)
        #Compute particle mass
        for i in range(0, self.numMeshElements):
            elementMass = self.density * self.restVolume[i]
            for particle in self.meshElements[i]:
                self.particleMass[particle] += (1.0/5.0) * elementMass
        #write points to file
        startTime = time.time() # start the timer
        self.latticeMeshObject.writeToFile(0, self.particles)
        self.timeMeasured.append(tuple(['writeToFile', time.time()-startTime])) # end the timer, add to the list
        #Extract stencil
        startTime = time.time() # start the timer
        self.precomputeStencilMatrix()
        self.timeMeasured.append(tuple(['preComputeStencilMatrix', time.time()-startTime])) # end the timer, add to the list
        self.recordOnce = True

    def printStencil(self):
        # for i in range(0, self.width+1):
        #     for j in range(0, self.height+1):
        #         for k in range(0, self.depth+1):
        #             print(str((i * (self.height + 1) * (self.depth + 1)) + (j * (self.depth + 1)) + k))
        #             print(self.stencil[i][j][k])
        return

    def precomputeStencilMatrix(self):
        #print(self.particleIndex.dataset)
        self.stencil = torch.zeros(self.width+1, self.height+1, self.depth+1, self.dimension, self.dimension, self.dimension)
        G = torch.transpose(self.GTranspose, 1, 2)
        GtG = torch.matmul(self.GTranspose, G)
        for i in range(0, self.numMeshElements):
            w = 2 * self.mu * self.restVolume[i] * GtG[i]
            #diagonals
            for j in range(0, self.dimension+1):
                self.stencil[self.meshElements[i][j][0], self.meshElements[i][j][1], self.meshElements[i][j][2], 1, 1, 1] += w[j][j]
            #upper
            for row in range(0, self.dimension+1):
                for col in range(row+1, self.dimension+1):
                    d = self.meshElements[i][col]-self.meshElements[i][row] + 1
                    self.stencil[self.meshElements[i][row][0], self.meshElements[i][row][1], self.meshElements[i][row][2], d[0], d[1], d[2]] += w[row][col]
                    d = self.meshElements[i][row]-self.meshElements[i][col] + 1
                    self.stencil[self.meshElements[i][col][0], self.meshElements[i][col][1], self.meshElements[i][col][2], d[0], d[1], d[2]] += w[row][col]
            # for row in range(0, self.dimension+1):
            #     for col in range(0, self.dimension+1):
            #         d = self.meshElements[i, col] - self.meshElements[i, row] + 1
            #         self.stencil[self.meshElements[i, row, 0], self.meshElements[i, row, 1], self.meshElements[i, row, 2], d[0], d[1], d[2]] += w[row][col]
        return

    def multiplyWithStiffnessMatrixPD(self, p, f):
        for i in range(0, self.numMeshElements):
            X = torch.zeros(size=[3, 4], dtype = torch.float32)
            for j in range(0, self.dimension+1):
                X[:, j] = p[:, self.meshElements[i][j][0], self.meshElements[i][j][1], self.meshElements[i][j][2]]
            deformationF = torch.mm(X, self.GTranspose[i])
            P = 2 * self.mu * deformationF
            Q = self.restVolume[i] * torch.mm(P, self.GTranspose[i].t())
            for j in range(0, 4):
                f[:, self.meshElements[i][j][0], self.meshElements[i][j][1], self.meshElements[i][j][2]] += Q[:, j]
        return f

    def multiplyWithStencil(self, x, q):
        startTime = time.time() # start the timer

        q[:, :, :, :] = 0.0
        xpad = torch.nn.functional.pad(x, (1,1,1,1,1,1), "constant", 0)
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
                        q[c, :, :, :] += xpad[c, x_start+di:x_end+di, y_start+dj:y_end+dj, z_start+dk:z_end+dk] * self.stencil[:, :, :, di, dj, dk]

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

    def computeElasticForce(self, forceTensor):
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
            P = 2 * self.mu * (deformationF - self.R[i])
            Q = -1 * self.restVolume[i] * torch.mm(P, self.GTranspose[i].t())
            for j in range(0, 4):
                forceTensor[:, self.meshElements[i][j][0], self.meshElements[i][j][1], self.meshElements[i][j][2]] += Q[:, j]
        return

    def resetConstrainedParticles(self, t, val):
        self.latticeMeshObject.resetConstrainedParticles(t, val)
        return

    def solveGlobalStep(self):
        rhs = torch.zeros(size=[self.dimension, self.width+1, self.height+1, self.depth+1], dtype=torch.float32)
        dx = torch.zeros(size= [self.dimension, self.width+1, self.height+1, self.depth+1], dtype=torch.float32)
        q = torch.zeros(size = [self.dimension, self.width+1, self.height+1, self.depth+1], dtype=torch.float32)
        s = torch.zeros(size = [self.dimension, self.width+1, self.height+1, self.depth+1], dtype=torch.float32)
        r = torch.zeros(size = [self.dimension, self.width+1, self.height+1, self.depth+1], dtype=torch.float32)

        self.computeElasticForce(rhs)
        self.resetConstrainedParticles(rhs, 0.0)
        #print("printing rhs")
        #print(rhs)
        
        cg = CGSolver(particles=dx, rhs=rhs, q=q, s=s, r=r, numIteration=60,
            minConvergenceNorm=1e-5, femObject=self)
        #solve
        cg.solve()
        dx = cg.getSolution()

        #update position vector with result
        # print(dx.shape)
        self.particles += dx

        return

    def pdSimulation(self):
        self.latticeMeshObject.setBoundaryConditions(self.particles,
            self.particleVelocity, self.stepEndTime)
        for i in range(0, 1):
            startTime = time.time() # start the timer
            self.solveLocalStep()
            self.timeMeasured.append(tuple(['solveLocalStep', time.time()-startTime])) # end the timer, add to the list
            startTime = time.time() # start the timer
            self.solveGlobalStep()
            self.timeMeasured.append(tuple(['solveGlobalStep', time.time()-startTime])) # end the timer, add to the list
        return

    def simulateFrame(self, frameNumber):
        print("frameNumber: "+ str(frameNumber))
        self.stepDt = self.frameDt / float(self.subSteps)
        for i in range(1, self.subSteps+1):
            self.stepEndTime = self.frameDt * (frameNumber-1) + self.stepDt * i
            if frameNumber == 5:
                r = (-2 * torch.rand(self.dimension, self.width+1, self.height+1, self.depth+1)) + 1.0
                self.particles += r
                self.latticeMeshObject.writeToFile(i, self.particles)
            self.pdSimulation()
            self.latticeMeshObject.writeToFile(i, self.particles)

    def getTimeMeasured(self):
        return self.timeMeasured   
