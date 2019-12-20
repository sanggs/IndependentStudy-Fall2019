import torch
import numpy as np

class CGSolver:
    def __init__(self, particles, rhs, q, s, r, numIteration, minConvergenceNorm, femObject):
        self.x = torch.from_numpy(particles)
        self.rhs = torch.from_numpy(rhs)
        self.q = torch.from_numpy(q)
        self.s = torch.from_numpy(s)
        self.r = torch.from_numpy(r)
        self.maxIterations = numIteration
        self.minConvergenceNorm = torch.tensor(minConvergenceNorm, dtype = torch.float32)
        self.femObject = femObject

    # def multiply(self, p, q):
    #     x = p.numpy()
    #     q = q.numpy()
    #     oneOverDtSquared = 1.0/(self.femObject.stepDt * self.femObject.stepDt)
    #     for i in range(0, self.femObject.numParticles):
    #         q[i] = self.femObject.particleMass[i] * oneOverDtSquared * x[i]
    #     # scale = 1.0 + self.femObject.rayleighCoefficient/self.femObject.stepDt
    #     # for i in range(0, self.femObject.numParticles):
    #     #     q[i] = 0.0
    #     q = torch.from_numpy(self.femObject.addProductWithStiffnessMatrixPD(x, q, 1.0))
    #     return q

    def multiplyWithA(self, p, q):
        x = p.numpy()
        q = q.numpy()
        q[:] = 0.0
        #y = np.zeros(shape=[self.femObject.numParticles, 1], dtype=np.float32)
        q = self.femObject.multiplyWithLHSPD(x, q)
        # q[:, 1] = self.femObject.multiplyWithLHSPD(x[:, 1], q[:, 1])
        # q[:, 2] = self.femObject.multiplyWithLHSPD(x[:, 2], q[:, 2])
        q = torch.from_numpy(q)
        return q

    def projectToZero(self, v):
        v = torch.from_numpy(self.femObject.resetConstrainedParticles(v.numpy(), 0.0))
        return v

    def getSolution(self):
        return self.x.numpy()

    def solve(self):
        self.q = self.multiplyWithA(self.x, self.q)
        self.r = self.rhs.sub(self.q)
        #print("Residue after 1st step")
        self.r = self.projectToZero(self.r)
        #print(self.r)
        convergenceNorm = 0
        # self.writeToFile(0)
        for i in range(0, self.maxIterations):
            convergenceNorm = torch.sqrt(torch.max(torch.sum(self.r*self.r, dim = 1)))
            print("printing convergence norm "+str(convergenceNorm))
            if convergenceNorm < self.minConvergenceNorm:
                print("Convergence Norm less than threshold")
                print(i)
                return
            if i > self.maxIterations:
                print("Ideally should not have come here")
                break
            rho = torch.sum(torch.sum(self.r*self.r, dim = 1))
            #print(rho)
            if i == 0:
                self.s = self.r
            else:
                self.s = ((rho/rhoOld) * self.s) + self.r
            self.q = self.multiplyWithA(self.s, self.q)
            self.q = self.projectToZero(self.q)
            #print("Q")
            #print(self.q)
            sDotq = torch.sum(torch.sum(self.s*self.q))
            #print("Iteration: "+str(i)+" sDotq: "+str(sDotq))
            if sDotq <= 0:
                print("CG matrix appears indefinite or singular, s_dot_q/s_dot_s="
                    +str(sDotq/(torch.sum(torch.sum(self.s*self.s)))))
            alpha = rho/sDotq
            self.x = alpha * self.s + self.x
            self.r = -alpha * self.q + self.r
            #print("R")
            #print(self.r)
            rhoOld = rho
        print("Ended after "+str(i)+ " iterations")
        print(convergenceNorm)
        return
