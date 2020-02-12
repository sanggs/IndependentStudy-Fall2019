import torch
import numpy as np
import sys

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
        self.particleIndex = torch.tensor(particleIndex, dtype=torch.float32)

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
            #print(self.restVolume[i])
        for i in range(0, self.numMeshElements):
            elementMass = self.density * self.restVolume[i]
            for particle in self.meshElements[i]:
                self.particleMass[particle] += (1.0/5.0) * elementMass
        self.latticeMeshObject.writeToFile(0, self.particles)
        #Extract stencil
        #self.constructLHSMatrix()

    def probeStencilMatrix(self):
        
        return
