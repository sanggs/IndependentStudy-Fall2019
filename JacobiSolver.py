import torch
import numpy as np

class JacobiSolver:
    def __init__(self, particles, femObject):
        self.particles = torch.from_numpy(particles)
        self.femObject = femObject
        self.dampingFactor = 2.0/3.0

    # def writeToFile(self, i):
    #     if i == 0:
    #         f = open("points.csv", "w")
    #     else:
    #         f = open("points.csv", "a")
    #     npArray = self.lattice.numpy()
    #     np.savetxt(f, npArray[None], delimiter=',')
    #     f.close()

    # def multiplyWithA(self):
    #     #print(self.gridWidth)
    #     #print(self.h)
    #     self.lattice.resize_((1,1,self.gridWidth, self.gridWidth))
    #     centralWeight = 4.0/(self.h*self.h)
    #     edgeWeight = -1.0/(self.h*self.h)
    #     mask = torch.tensor([[0,edgeWeight,0],[edgeWeight,centralWeight,edgeWeight],[0,edgeWeight,0]], dtype=torch.float32)
    #     mask.resize_((1,1,3,3))
    #     output = torch.nn.functional.conv2d(self.lattice, mask, bias=None, stride=1, padding=0)
    #     b = torch.nn.functional.pad(output, (1,1,1,1), mode='constant', value=(-self.b/(self.h * self.h)))
    #     self.lattice = self.lattice.resize_((self.gridWidth*self.gridWidth))
    #     return b.view((self.gridWidth*self.gridWidth))

    # def getResidual(self, rhs):
    #     q = self.multiplyWithA()
    #     #print("q from get Residual")
    #     #print(q.view((self.gridWidth, self.gridWidth)))
    #     residue = rhs.sub(q)
    #     #print("residue from getResidual")
    #     self.projectToZero(residue)
    #     #print(residue.view((self.gridWidth, self.gridWidth)))
    #     return residue

    # def projectToZero(self, v):
    #     v = v.resize_((self.gridWidth, self.gridWidth))
    #     v[0,:] = 0
    #     v[self.gridWidth-1,:] = 0
    #     v[:,0] = 0
    #     v[:,self.gridWidth-1] = 0
    #     v = v.resize_((self.gridWidth*self.gridWidth))

    def dampedJacobi(self, b, q, dInverse, r, num_iteration, residue):
        self.residue = torch.tensor(residue, dtype = torch.float32)
        self.maxIterations = num_iteration
        q = self.multiplyWithA()
        #print("printing q")
        #print(q.view(self.gridWidth, self.gridWidth))
        r = b.sub(q)
        #print("Residue after 1st step")
        self.projectToZero(r)
        print(r.view((self.gridWidth, self.gridWidth)))
        #print(b.shape)
        convergence_norm = 0
        self.writeToFile(0)
        for i in range(0, self.maxIterations):
            convergence_norm = torch.sqrt(torch.max(r*r))
            print("printing convergence norm "+str(convergence_norm))
            #print(convergence_norm)
            #if convergence_norm < self.residue:
            #print("Convergence Norm less than threshold")
            #print(i)
            #return
            if i > self.maxIterations:
                #print("printing convergence norm")
                #print(convergence_norm)
                print("Ideally should not have come here")
                break
            r = dInverse * r * self.dampingFactor
            self.projectToZero(r)
            self.lattice = self.lattice + r
            #print("printing lattice after "+ str(i+1))
            #print(self.lattice)
            q = self.multiplyWithA()
            r = b.sub(q)
            self.projectToZero(r)
            self.writeToFile(i+1)
        #print("Ended after "+str(i)+ " iterations")
        print(convergence_norm)
        return
