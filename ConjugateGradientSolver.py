import torch
import numpy as np
import sys
class CGSolver:
    def __init__(self, particles, rhs, q, s, r, numIteration, minConvergenceNorm, width, height, depth, femObject):
        self.x = particles
        self.rhs = rhs
        self.q = q
        self.s = s
        self.r = r
        self.width = width
        self.height = height 
        self.depth = depth
        self.maxIterations = numIteration
        self.minConvergenceNorm = torch.tensor(minConvergenceNorm, dtype = torch.float32)
        self.femObject = femObject

    def multiplyWithA(self, p, q):
        self.femObject.multiplyWithStencil(p, q)
        return

    def projectToZero(self, v):
        self.femObject.resetConstrainedParticles(v, 0.0)
        return

    def getSolution(self):
        return self.x

    def solve(self):
        self.multiplyWithA(self.x, self.q)
        self.r = self.rhs.sub(self.q)
        self.projectToZero(self.r)
        # print("Residue after 1st step")
        # print(self.r)
        convergenceNorm = 0
        for i in range(0, self.maxIterations):
            convergenceNorm = torch.sqrt(torch.max(torch.sum(self.r*self.r, dim = 0)))
            # print("printing convergence norm "+str(convergenceNorm))
            if convergenceNorm < self.minConvergenceNorm:
                print("Convergence Norm less than threshold")
                print(i)
                return
            if i > self.maxIterations:
                print("Ideally should not have come here")
                break
            rho = torch.sum(self.r*self.r)
            if i == 0:
                self.s[:, 1:self.width+2, 1:self.height+2, 1:self.depth+2] = self.r
            else:
                self.s[:, 1:self.width+2, 1:self.height+2, 1:self.depth+2] = ((rho/rhoOld) * self.s[:, 1:self.width+2, 1:self.height+2, 1:self.depth+2]) + self.r
            self.multiplyWithA(self.s, self.q)
            self.projectToZero(self.q)
            
            sDotq = torch.sum(self.s[:, 1:self.width+2, 1:self.height+2, 1:self.depth+2]*self.q)
            # print("Iteration: "+str(i)+" sDotq: "+str(sDotq))
            if sDotq <= 0:
                print("CG matrix appears indefinite or singular, s_dot_q/s_dot_s="
                    +str(sDotq/(torch.sum(self.s*self.s))))
            alpha = rho/sDotq
            self.x[:, 1:self.width+2, 1:self.height+2, 1:self.depth+2] = alpha * self.s[:, 1:self.width+2, 1:self.height+2, 1:self.depth+2] + self.x[:, 1:self.width+2, 1:self.height+2, 1:self.depth+2]
            self.r = -alpha * self.q + self.r
            rhoOld = rho
            
        print("Ended after "+str(i)+ " iterations")
        print(convergenceNorm)
        return
