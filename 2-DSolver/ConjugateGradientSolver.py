import torch
import numpy as np

class CGSolver:
    def __init__(self, particles, nodeType, gridWidth, b, h, rhs, q, s, r, numIteration, minConvergenceNorm):
        self.x = particles
        self.nodeType = nodeType
        self.gridWidth = gridWidth
        self.b = b
        self.h = h
        self.rhs = rhs
        self.q = q
        self.s = s
        self.r = r
        self.maxIterations = numIteration
        self.minConvergenceNorm = torch.tensor(minConvergenceNorm, dtype = torch.float32)

    def multiplyWithA(self, p, q):
        p.resize_((1,1,self.gridWidth, self.gridWidth))
        cornerWeight = 0.0/(4.0*self.h*self.h)
        centralWeight = 4.0/(self.h*self.h)
        edgeWeight = -1.0/(self.h*self.h)
        mask = torch.tensor([[cornerWeight,edgeWeight,cornerWeight],[edgeWeight,centralWeight,edgeWeight],[cornerWeight,edgeWeight,cornerWeight]], dtype=torch.float32)
        mask.resize_((1,1,3,3))
        output = torch.nn.functional.conv2d(p, mask, bias=None, stride=1, padding=1)
        #q = torch.nn.functional.pad(output, (1,1,1,1), mode='constant', value=(-self.b/(self.h * self.h)))
        q = output
        p = p.resize_((self.gridWidth*self.gridWidth))
        return q.resize_((self.gridWidth*self.gridWidth))

    def projectToZero(self, v):
        v.resize_((self.gridWidth, self.gridWidth))
        self.nodeType.resize_((self.gridWidth, self.gridWidth))
        v = v * self.nodeType
        self.nodeType.resize_((self.gridWidth * self.gridWidth))
        return v.resize_((self.gridWidth * self.gridWidth))

    def getSolution(self):
        return self.x

    def getResidual(self):
        self.q = self.multiplyWithA(self.x, self.q)
        self.r = self.rhs.sub(self.q)
        self.r = self.projectToZero(self.r)
        return self.r

    def solve(self):
        self.q = self.multiplyWithA(self.x, self.q)
        self.r = self.rhs.sub(self.q)
        self.r = self.projectToZero(self.r)
        convergenceNorm = 0
        for i in range(0, self.maxIterations):
            convergenceNorm = torch.sqrt(torch.max(torch.sum(self.r*self.r)))

            print("printing convergence norm "+str(convergenceNorm) + " i: "+str(i))
            if convergenceNorm < self.minConvergenceNorm:
                print("Convergence Norm less than threshold")
                print(i)
                return
            if i > self.maxIterations:
                print("Ideally should not have come here")
                break
            rho = torch.sum(torch.sum(self.r*self.r))
            if i == 0:
                self.s = self.r
            else:
                self.s = ((rho/rhoOld) * self.s) + self.r
            self.q = self.multiplyWithA(self.s, self.q)
            self.q = self.projectToZero(self.q)
            #print("Q")
            sDotq = torch.sum(torch.sum(self.s*self.q))
            #print("Iteration: "+str(i)+" sDotq: "+str(sDotq))
            if sDotq <= 0:
                print("CG matrix appears indefinite or singular, s_dot_q/s_dot_s="
                    +str(sDotq/(torch.sum(torch.sum(self.s*self.s)))))
            alpha = rho/sDotq
            self.x += alpha * self.s
            self.r += -alpha * self.q
            rhoOld = rho
        #print("Ended after "+str(i)+ " iterations")
        print(convergenceNorm)
        return
