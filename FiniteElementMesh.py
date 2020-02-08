import torch
import numpy as np
import sys

from ConjugateGradientSolver import CGSolver

class FiniteElementMesh:
    def __init__(self, density, mu, lmbda, rayleighCoefficient, frames, frameDt, subSteps=1):
        self.density = density
        self.mu = mu
        self.lmbda = lmbda
        self.rayleighCoefficient = rayleighCoefficient
        self.nFrames = frames
        self.subSteps = subSteps
        self.frameDt = frameDt
        self.particleMass = None
        self.particleVelocity = None
        self.dMInverse = None
        self.restVolume = None
        self.particles = None
        self.meshElements = None
        self.numParticles = None
        self.numMeshElements = None

    def setParticles(self, particles, width, height, depth, particleIndex):
        self.particles = np.asarray(particles, dtype=np.float32)
        self.numParticles = len(particles)
        self.width = width
        self.height = height
        self.depth = depth
        self.particleIndex = particleIndex

    def setMeshElements(self, meshElements):
        self.meshElements = np.asarray(meshElements, dtype=np.int)
        self.numMeshElements = len(meshElements)

    def setHandles(self, leftHandle, rightHandle):
        if len(leftHandle) == 0 or len(rightHandle) == 0:
            print('Left and Right handles not set')
            sys.exit()
        self.leftHandleIndices = leftHandle
        self.rightHandleIndices = rightHandle

    def registerLatticeMeshObject(self, lm):
        self.latticeMeshObject = lm

    def initialiseUndeformedConfiguration(self):
        if self.particles is None or self.meshElements is None:
            print("Particles and meshElements not set")
            sys.exit()
        self.particleVelocity = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
        dM = np.zeros(shape = [self.numMeshElements, 3, 3], dtype=np.float32)
        self.dMInverse = np.zeros(shape = [self.numMeshElements, 3, 3], dtype=np.float32)
        #rint(dM.shape)
        self.restVolume = np.zeros(self.numMeshElements, dtype=np.float32)
        self.particleMass = np.zeros(self.numParticles, dtype=np.float32)
        for i in range(0, self.numMeshElements):
            #we know meshElements will have 4 elements
            p0 = self.meshElements[i][0]
            for j in range(1, 4):
                pj = self.meshElements[i][j]
                dM[i, :, j-1] = self.particles[pj] - self.particles[p0]
            self.dMInverse[i] = np.linalg.inv(dM[i])
            self.restVolume[i] = 0.5 * np.linalg.det(dM[i])
            #print(self.restVolume[i])
            elementMass = self.density * self.restVolume[i]
            for particle in self.meshElements[i]:
                self.particleMass[particle] += (1.0/5.0) * elementMass
        self.latticeMeshObject.writeToFile(0, self.particles)
        self.constructLHSMatrix()

    def resetConstrainedParticles(self, v, value):
        v = self.latticeMeshObject.resetConstrainedParticles(v, value)
        return v

    def addElasticForce(self, forceVector):
        for i in range(0, self.numMeshElements):
            dS = np.zeros(shape = [3, 3], dtype=np.float32)
            p0 = self.meshElements[i][0]
            for j in range(1, 4):
                pj = self.meshElements[i][j]
                dS[:, j-1] = self.particles[pj] - self.particles[p0]
            #deformationMatrix
            deformationF = np.matmul(dS, self.dMInverse[i])
            strain = (0.5 * (deformationF + np.transpose(deformationF))) - np.identity(3, dtype=np.float32)
            P = (2. * self.mu * strain) + (self.lmbda * np.trace(strain) * np.identity(3, dtype=np.float32))
            H = (-1.0 * self.restVolume[i]) * np.matmul(P, np.transpose(self.dMInverse[i]))
            for j in range(1,4):
                forceVector[self.meshElements[i][j]] += H[:, j-1]
                forceVector[self.meshElements[i][0]] -= H[:, j-1]
        return forceVector

    def addProductWithStiffnessMatrix(self, pos, forceVector, scale):
        for i in range(0, self.numMeshElements):
            dS = np.zeros(shape = [3, 3], dtype=np.float32)
            p0 = self.meshElements[i][0]
            for j in range(1, 4):
                pj = self.meshElements[i][j]
                dS[:, j-1] = pos[pj] - pos[p0]
            #deformationMatrix
            #print(self.dMInverse[i])
            deformationF = np.matmul(dS, self.dMInverse[i])
            strain_rate = 0.5 * (deformationF + np.transpose(deformationF))
            P_damping = scale * (2. * self.mu * strain_rate + self.lmbda * np.trace(strain_rate) * np.identity(3, dtype=np.float32))
            H_damping = self.restVolume[i] * np.matmul(P_damping, np.transpose(self.dMInverse[i]))
            for j in range(1,4):
                forceVector[self.meshElements[i][j]] += H_damping[:, j-1]
                forceVector[self.meshElements[i][0]] -= H_damping[:, j-1]
        return forceVector

    def simulateSubstep(self):
        print("Going to call multiGrid solver from here")

        #save previous velocity
        lastV = self.particleVelocity

        # // Construct initial guess for next-timestep
        # //   Velocities -> Same as last timestep
        # //   Positions -> Using Forward Euler
        for i in range(0, self.numParticles):
            self.particles[i] += self.stepDt * self.particleVelocity[i]

        #pull handles left and right
        self.latticeMeshObject.setBoundaryConditions(
            self.particles, self.particleVelocity, self.stepEndTime)

        #initialise all vectors for solver
        rhs = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
        dx = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
        q = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
        s = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
        r = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)

        #compute forces because of the push and pull
        rhs = self.addElasticForce(rhs)
        for i in range(0, self.numParticles):
            rhs[i] += (self.particleMass[i]/self.stepDt) * (lastV[i] - self.particleVelocity[i])
        rhs = self.addProductWithStiffnessMatrix(self.particleVelocity, rhs, -self.rayleighCoefficient)
        # reset forces for constrained particles (the ones on the handles)
        rhs = self.resetConstrainedParticles(rhs, 0.0)
        #print(rhs)
        #initialise CG Solver
        cg = CGSolver(particles=dx, rhs=rhs, q=q, s=s, r=r, numIteration=10,
            minConvergenceNorm=1e-5, femObject=self)
        #solve
        cg.solve()
        dx = cg.getSolution()

        #update position vector with result
        for i in range(0, self.numParticles):
            self.particles[i] += dx[i]
            self.particleVelocity[i] += (1/self.stepDt) * dx[i];
        self.latticeMeshObject.writeToFile(1, self.particles)

    def solveLocalStep(self):
        self.minRMatrix = np.zeros(shape = [self.numMeshElements, 3, 3], dtype=np.float32)
        for i in range(0, self.numMeshElements):
            dS = np.zeros(shape = [3, 3], dtype=np.float32)
            p0 = self.meshElements[i][0]
            for j in range(1, 4):
                pj = self.meshElements[i][j]
                dS[:, j-1] = self.particles[pj] - self.particles[p0]
            #deformationMatrix
            deformationF = np.matmul(dS, self.dMInverse[i])
            u, sigma, vTranspose = np.linalg.svd(deformationF, full_matrices=True)
            if np.linalg.det(u) < 0:
                if np.linalg.det(vTranspose) < 0:
                    u[:, 2] = -1.0 * u[:, 2]
                    vTranspose[2, :] = -1.0 * vTranspose[2, :]
                else:
                    u[:, 2] = -1.0 * u[:, 2]
                    sigma[2] = -1.0 * sigma[2]
            elif np.linalg.det(vTranspose) < 0:
                vTranspose[2, :] = -1.0 * vTranspose[2, :]
                sigma[2] = -1.0 * sigma[2]
            self.minRMatrix[i] = np.matmul(u, vTranspose)

    def addElasticForcePD(self, forceVector):
        for i in range(0, self.numMeshElements):
            dS = np.zeros(shape = [3, 3], dtype=np.float32)
            p0 = self.meshElements[i][0]
            for j in range(1, 4):
                pj = self.meshElements[i][j]
                dS[:, j-1] = self.particles[pj] - self.particles[p0]
            #deformationMatrix
            deformationF = np.matmul(dS, self.dMInverse[i])
            u, sigma, vTranspose = np.linalg.svd(deformationF, full_matrices=True)

            if np.linalg.det(u) < 0:
                if np.linalg.det(vTranspose) < 0:
                    u[:, 2] = -1.0 * u[:, 2]
                    vTranspose[2, :] = -1.0 * vTranspose[2, :]
                    deformationF = np.matmul(u, np.matmul(np.diag(sigma), vTranspose))
                else:
                    u[:, 2] = -1.0 * u[:, 2]
                    sigma[2] = -1.0 * sigma[2]
                    deformationF = np.matmul(u, np.matmul(np.diag(sigma), vTranspose))
            elif np.linalg.det(vTranspose) < 0:
                vTranspose[2, :] = -1.0 * vTranspose[2, :]
                sigma[2] = -1.0 * sigma[2]
                deformationF = np.matmul(u, np.matmul(np.diag(sigma), vTranspose))

            P = 2*self.mu*(deformationF - self.minRMatrix[i])
            H = (-1.0 * self.restVolume[i]) * np.matmul(P, np.transpose(self.dMInverse[i]))
            for j in range(1,4):
                forceVector[self.meshElements[i][j]] += H[:, j-1]
                forceVector[self.meshElements[i][0]] -= H[:, j-1]
        return forceVector

    def multiplyWithStiffnessMatrixPD(self, dx, forceVector):
        for i in range(0, self.numMeshElements):
            dS = np.zeros(shape = [3, 3], dtype=np.float32)
            p0 = self.meshElements[i][0]
            for j in range(1, 4):
                pj = self.meshElements[i][j]
                dS[:, j-1] = dx[pj] - dx[p0]
            #deformationMatrix
            deformationF = np.matmul(dS, self.dMInverse[i])

            P = 2 * self.mu * deformationF
            H = self.restVolume[i] * np.matmul(P, np.transpose(self.dMInverse[i]))
            for j in range(1,4):
                forceVector[self.meshElements[i][j]] += H[:, j-1]
                forceVector[self.meshElements[i][0]] -= H[:, j-1]
        return forceVector

    def prepareLHS(self):
        self.stencilMatrix = np.zeros(shape=[self.numParticles,3,3,3], dtype=np.float32)
        for i in range(0, self.width+1):
            for j in range(0, self.height+1):
                for k in range(0, self.depth+1):
                    pIndex = self.gridToParticleID(i, j, k)
                    p = self.findParticlePos(pIndex)
                    for v in range(0,1):
                        Ai = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
                        output = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
                        Ai[p][v] = 1.0
                        #Now multiplyWithStiffnessMatrixPD
                        output = self.multiplyWithStiffnessMatrixPD(Ai, output)
                        #Collect Stencil weights
                        for x in range(-1, 2):
                            for y in range(-1, 2):
                                for z in range(-1, 2):
                                    pIndex1 = self.gridToParticleID(i+x, j+y, k+z)
                                    if pIndex1 >= 0 and pIndex1 < self.numParticles:
                                        p1 = self.findParticlePos(pIndex1)
                                        self.stencilMatrix[p][x+1][y+1][z+1] = output[p1][v]
        for i in range(0, self.numParticles):
            print(self.stencilMatrix[i])

    def constructLHSMatrix(self):
        self.LHSMatrix = np.zeros(shape=[self.numParticles,self.numParticles], dtype=np.float32)
        for i in range(0, self.width+1):
            for j in range(0, self.height+1):
                for k in range(0, self.depth+1):
                    pIndex = self.gridToParticleID(i, j, k)
                    p = self.findParticlePos(pIndex)
                    for v in range(0,1):
                        Ai = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
                        output = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
                        Ai[p][v] = 1.0
                        #Now multiplyWithStiffnessMatrixPD
                        output = self.multiplyWithStiffnessMatrixPD(Ai, output)
                        #Collect Stencil weights
                        for x in range(-1, 2):
                            for y in range(-1, 2):
                                for z in range(-1, 2):
                                    pIndex1 = self.gridToParticleID(i+x, j+y, k+z)
                                    if pIndex1 >= 0 and pIndex1 < self.numParticles:
                                        p1 = self.findParticlePos(pIndex1)
                                        self.LHSMatrix[p][p1] = output[p1][v]
        for i in range(0, self.numParticles):
            print(self.LHSMatrix[i])

    def findParticlePos(self, pIndex):
        for i in range(0, self.numParticles):
            if self.particleIndex[i] == pIndex:
                return i

    def gridToParticleID(self, i, j , k):
        return ((i) * (self.height+1) * (self.depth+1)) + ((j) * (self.depth+1)) + k

    def multiplyWithStencil(self, input, output):
        for axis in range(0, 3):
            for i in range(0, self.width+1):
                for j in range(0, self.height+1):
                    for k in range(0, self.depth+1):
                        p = self.gridToParticleID(i, j, k)
                        currentP = self.findParticlePos(p)
                        newParticleVal = 0.0
                        for x in range(-1, 2):
                            for y in range(-1, 2):
                                for z in range(-1, 2):
                                    pIndex1 = self.gridToParticleID(i+x, j+y, k+z)
                                    if pIndex1 >= 0 and pIndex1 < self.numParticles:
                                        p1 = self.findParticlePos(pIndex1)
                                        newParticleVal += self.stencilMatrix[currentP][x+1][y+1][z+1] * input[p1][axis]
                        output[currentP][axis] = newParticleVal
        return output

    def multiplyWithLHSMatrix(self, x, y):
        y += np.matmul(self.LHSMatrix, x)
        return y

    def multiplyWithLHSPD(self, x, y):
        y = self.multiplyWithLHSMatrix(x, y)
        # y = self.multiplyWithStencil(x, y)
        # y = self.multiplyWithStiffnessMatrixPD(x, y)
        return y

    '''
    def multiplyWithRHSPD(self):
        rhs = self.RHS
        p = self.minRMatrix.reshape((3*self.numMeshElements, 3))
        y = np.matmul(rhs, p)
        return y
    '''

    def solveGlobalStep(self):
        print("Solving global step")

        #pull handles left and right
        self.latticeMeshObject.setBoundaryConditions(
            self.particles, self.particleVelocity, self.stepEndTime)

        #initialise all vectors for solver
        rhs = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
        dx = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
        q = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
        s = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)
        r = np.zeros(shape=[self.numParticles, 3], dtype=np.float32)

        #compute forces because of the push and pull
        rhs = self.addElasticForcePD(rhs)
        rhs = self.resetConstrainedParticles(rhs, 0.0)

        #initialise CG Solver
        cg = CGSolver(particles=dx, rhs=rhs, q=q, s=s, r=r, numIteration=50,
            minConvergenceNorm=1e-5, femObject=self)
        #solve
        cg.solve()
        dx = cg.getSolution()

        #update position vector with result
        for i in range(0, self.numParticles):
            self.particles[i] += dx[i]

    def projectiveDynamicsSimulation(self):
        for i in range(0, 1):
            self.solveLocalStep()
            self.solveGlobalStep()

    def simulateFrame(self, frameNumber):
        self.stepDt = self.frameDt / float(self.subSteps)
        for i in range(1, self.subSteps+1):
            self.stepEndTime = self.frameDt * (frameNumber-1) + self.stepDt * i
            if frameNumber == 5:
                r = (-2 * np.random.rand(self.numParticles,3)) + 1.0
                self.particles += r
                self.latticeMeshObject.writeToFile(1, self.particles)
            self.projectiveDynamicsSimulation()
            self.latticeMeshObject.writeToFile(1, self.particles)
