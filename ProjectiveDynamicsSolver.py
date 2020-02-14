import torch
import numpy as np
import sys

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
        self.particleMass = None
        self.particleVelocity = None
        self.dMInverse = None
        self.restVolume = None
        self.setParticles(particles, particleIndex)
        self.setMeshElements(meshElements)
        self.latticeMeshObject = None
        if(lm):
            self.registerLatticeMeshObject(lm)
        self.initialiseUndeformedConfiguration()

    def setParticles(self, particles, particleIndex):
        self.particles = torch.tensor(particles, dtype=torch.float32)
        self.numParticles = len(particles)
        print(self.particles.shape)
        self.particleIndex = torch.utils.data.DataLoader(particleIndex)

    def setMeshElements(self, meshElements):
        self.meshElements = torch.tensor(meshElements, dtype = torch.int64)
        self.numMeshElements = len(meshElements)

    def registerLatticeMeshObject(self, lm):
        self.latticeMeshObject = lm

    def initialiseUndeformedConfiguration(self):
        if self.particles is None or self.meshElements is None:
            print("Particles and meshElements not set")
            sys.exit()#sanity check for particles and meshElements
        # Builder Matrix: needed to compute dM from X
        builder = torch.zeros(size=[4,3], dtype = torch.float32)
        builder[1:4, :] = torch.eye(3)
        builder[0, :] = -1 * torch.ones(3)
        #Velocity
        self.particleVelocity = torch.zeros([self.numParticles, 3], dtype=torch.float32)
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
            for j in range(0, 4):
                X[i, :, j] = self.particles[self.meshElements[i][j]]
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
        self.latticeMeshObject.writeToFile(0, self.particles)
        #Extract stencil
        self.precomputeStencilMatrix()

    def precomputeStencilMatrix(self):
        print(self.particleIndex.dataset)
        self.stencil = torch.zeros(self.numParticles, self.numParticles)
        G = torch.transpose(self.GTranspose, 1, 2)
        GtG = torch.matmul(self.GTranspose, G)
        for i in range(0, self.numMeshElements):
            w = 2 * self.mu * self.restVolume[i] * GtG[i]
            for row in range(0, 4):
                pRow = self.meshElements[i][row]
                for col in range(row, 4):
                    pCol = self.meshElements[i][col]
                    self.stencil[pRow][pCol] += w[row][col]
                    self.stencil[pCol][pRow] += w[col][row]
        return

    def getBuilderTensor(self):
        builder = torch.zeros(size=[4,3], dtype = torch.float32)
        builder[1:4, :] = torch.eye(3)
        builder[0, :] = -1 * torch.ones(3)
        return builder

    def multiplyWithStencil(self, x, q):
        q += torch.matmul(self.stencil, x)
        return q

    def computeDs(self, elementNum):
        builder = self.getBuilderTensor()
        X = torch.zeros(size=[3, 4], dtype = torch.float32)
        for j in range(0, 4):
            X[:, j] = self.particles[self.meshElements[elementNum][j]]
        return torch.matmul(X, builder)

    def solveLocalStep(self):
        self.R = torch.zeros([self.numMeshElements, 3, 3], dtype = torch.float32)
        # Builder Matrix: needed to compute dM from X
        for i in range(0, self.numMeshElements):
            # dS = self.computeDs(i)
            X = torch.zeros(size=[3, 4], dtype = torch.float32)
            for j in range(0, 4):
                X[:, j] = self.particles[self.meshElements[i][j]]
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
            for j in range(0, 4):
                X[:, j] = self.particles[self.meshElements[i][j]]
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
                forceTensor[self.meshElements[i][j]] += Q[:, j]
        return

    def resetConstrainedParticles(self, t, val):
        t = torch.from_numpy(self.latticeMeshObject.resetConstrainedParticles(t.numpy(), val))
        return t

    def solveGlobalStep(self):
        rhs = torch.zeros(size=[self.numParticles, 3], dtype=torch.float32)
        dx = torch.zeros(size=[self.numParticles, 3], dtype=torch.float32)
        q = torch.zeros(size=[self.numParticles, 3], dtype=torch.float32)
        s = torch.zeros(size=[self.numParticles, 3], dtype=torch.float32)
        r = torch.zeros(size=[self.numParticles, 3], dtype=torch.float32)

        self.computeElasticForce(rhs)
        rhs = self.resetConstrainedParticles(rhs, 0.0)
        #print(rhs)

        cg = CGSolver(particles=dx, rhs=rhs, q=q, s=s, r=r, numIteration=50,
            minConvergenceNorm=1e-5, femObject=self)
        #solve
        cg.solve()
        dx = cg.getSolution()

        #update position vector with result
        for i in range(0, self.numParticles):
            self.particles[i] += dx[i]

        return

    def pdSimulation(self):
        self.latticeMeshObject.setBoundaryConditions(self.particles,
            self.particleVelocity, self.stepEndTime)
        self.solveLocalStep()
        self.solveGlobalStep()
        return

    def simulateFrame(self, frameNumber):
        self.stepDt = self.frameDt / float(self.subSteps)
        for i in range(1, self.subSteps+1):
            self.stepEndTime = self.frameDt * (frameNumber-1) + self.stepDt * i
            if frameNumber == 5:
                r = (-2 * torch.rand(self.numParticles,3)) + 1.0
                self.particles += r
                self.latticeMeshObject.writeToFile(1, self.particles.numpy())
            self.pdSimulation()
            self.latticeMeshObject.writeToFile(1, self.particles.numpy())
