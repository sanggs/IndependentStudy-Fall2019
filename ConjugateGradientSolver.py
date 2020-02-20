import torch
import numpy as np
import sys
class CGSolver:
    def __init__(self, particles, rhs, q, s, r, numIteration, minConvergenceNorm, femObject):
        self.x = particles
        self.rhs = rhs
        self.q = q
        self.s = s
        self.r = r
        # self.x = torch.from_numpy(particles)
        # self.rhs = torch.from_numpy(rhs)
        # self.q = torch.from_numpy(q)
        # self.s = torch.from_numpy(s)
        # self.r = torch.from_numpy(r)
        self.maxIterations = numIteration
        self.minConvergenceNorm = torch.tensor(minConvergenceNorm, dtype = torch.float32)
        self.femObject = femObject

    def multiplyWithA(self, p, q):
        q = self.femObject.multiplyWithStencil(p, q)
        return q

    def projectToZero(self, v):
        v = self.femObject.resetConstrainedParticles(v, 0.0)
        return v

    def getSolution(self):
        return self.x

    def solve(self):
        self.q = self.multiplyWithA(self.x, self.q)
        self.r = self.rhs.sub(self.q)
        # print("Residue after 1st step")
        self.r = self.projectToZero(self.r)
        # print(self.r)
        convergenceNorm = 0
        # self.writeToFile(0)
        for i in range(0, self.maxIterations):
            convergenceNorm = torch.sqrt(torch.max(torch.sum(self.r*self.r, dim = 3)))
            print("printing convergence norm "+str(convergenceNorm))
            if convergenceNorm < self.minConvergenceNorm:
                print("Convergence Norm less than threshold")
                print(i)
                return
            if i > self.maxIterations:
                print("Ideally should not have come here")
                break
            rho = torch.sum(self.r*self.r)
            #print(rho)
            if i == 0:
                self.s = self.r
            else:
                self.s = ((rho/rhoOld) * self.s) + self.r
            self.q = self.multiplyWithA(self.s, self.q)
            self.q = self.projectToZero(self.q)
            
            #print("Q")
            #print(self.q)
            sDotq = torch.sum(self.s*self.q)
            print("Iteration: "+str(i)+" sDotq: "+str(sDotq))
            if sDotq <= 0:
                print("CG matrix appears indefinite or singular, s_dot_q/s_dot_s="
                    +str(sDotq/(torch.sum(self.s*self.s))))
            alpha = rho/sDotq
            self.x = alpha * self.s + self.x
            self.r = -alpha * self.q + self.r
            #print("R")
            #print(self.r)
            rhoOld = rho
            
        print("Ended after "+str(i)+ " iterations")
        print(convergenceNorm)
        return
