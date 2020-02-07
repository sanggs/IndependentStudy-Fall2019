import torch
import numpy as np
import math
import numba
from JacobiSolver import JacobiSolver

class LaplacianSolver:
    def __init__(self, lattice, nodeType, gridWidth):
        self.lattice = lattice
        self.nodeType = nodeType
        self.gridWidth = gridWidth
        self.h = gridWidth-1
        self.centralWeight = 4.0
        self.edgeWeight = 1.0
        self.b = 0.0

    def sampleFunction(self):
        print(self.lattice.dtype)

    def writeToFile(self, i):
        if i == 0:
            f = open("points1.csv", "w")
        else:
            f = open("points1.csv", "a")
        npArray = self.lattice.numpy()
        np.savetxt(f, npArray[None], delimiter=',')
        f.close()

    def projectToZero(self, v, nodeType, gW):
        v = v.resize_((gW, gW))
        v[0,:] = 0
        v[gW-1,:] = 0
        v[:,0] = 0
        v[:,gW-1] = 0
        v = v.resize_((gW*gW))

    # def projectToZero(self, v):
    #     v = v.resize_((self.gridWidth, self.gridWidth))
    #     v[0,:] = 0
    #     v[self.gridWidth-1,:] = 0
    #     v[:,0] = 0
    #     v[:,self.gridWidth-1] = 0
    #     v = v.resize_((self.gridWidth*self.gridWidth))

    def smoother(self, v, gW, h, b, iter=3):
        v.resize_((1,1,gW, gW))
        for i in range(0,iter):
            centralWeight = 4.0/(h * h)
            edgeWeight = -1.0/(h * h)
            mask = torch.tensor([[0,edgeWeight,0],[edgeWeight,centralWeight,edgeWeight],[0,edgeWeight,0]], dtype=torch.float32)
            mask.resize_((1,1,3,3))
            output = torch.nn.functional.conv2d(v, mask, bias=None, stride=1, padding=0)
            padded_output = torch.nn.functional.pad(output, (1,1,1,1), mode='constant', value=(-b/(h * h)))
            print("smoother")
            #print(padded_output)
            v = padded_output
        return padded_output.view((gW*gW))

    def jacobiSolver(self):
        len_h = self.gridWidth * self.gridWidth

        rhs_h = torch.zeros((len_h), dtype=torch.float32)
        rhs_h = rhs_h + (-self.b/(self.h * self.h))
        #print(rhs_h)
        self.projectToZero(rhs_h, self.nodeType, self.gridWidth)

        q_h = torch.zeros((len_h), dtype=torch.float32)

        #Set the boundary
        self.setBoundary(v=self.lattice, nT=self.nodeType, gW=self.gridWidth, value=0)
        #self.lattice.resize_((len_h))

        dInverse_h = torch.zeros((len_h), dtype = torch.float32)
        dInverse_h = dInverse_h + (self.h * self.h)/(self.centralWeight)
        dInverse_h[0] = self.h * self.h
        dInverse_h[len_h-1] = self.h * self.h
        self.writeToFile(0)
        residue_h = torch.zeros((len_h), dtype=torch.float32)

        #self.lattice = jb_h.lattice
        # residue_h = jb_h.getResidual(rhs_h)
        # print(torch.sqrt(torch.max(residue_h*residue_h)))
        for i in range(1, 11, 1):
            #Do it for 2h grid and then call solve
            len_2h = int(self.gridWidth/2 * self.gridWidth/2)
            gridWidth_2h = int(self.gridWidth/2)
            h_2h = gridWidth_2h - 1;

            jb_h = JacobiSolver(lattice=self.lattice, nodeType=self.nodeType, gridWidth=self.gridWidth, b=self.b)
            jb_h.dampedJacobi(rhs_h, q_h, dInverse_h, residue_h, 1, 1e-5)
            self.lattice = jb_h.lattice
            residue_h = jb_h.getResidual(rhs=rhs_h)
            #self.writeToFile(i)
            #residue_h = self.smoother(residue_h, self.gridWidth, self.h, self.b)
            print("printing residue_h")
            print(residue_h)

            lattice_2h = torch.zeros((len_2h), dtype=torch.float32)

            #nodeType
            nodeType_h = self.nodeType.resize_((1,1,self.gridWidth, self.gridWidth))
            maxpool = torch.nn.MaxPool2d((2,2), stride=2)
            nodeType_2h = maxpool(nodeType_h)
            nodeType_2h.resize_((len_2h))

            #rhs => residue
            wCorner = 1/16
            wEdge = 3/16
            wCenter = 9/16
            print("residue_h")
            #print(residue_h.view((self.gridWidth, self.gridWidth)))
            residue_h = residue_h.resize_((1,1,self.gridWidth, self.gridWidth))
            residue_h = -1 * residue_h
            print(residue_h)
            mask = torch.tensor([[wCorner, wEdge, wEdge, wCorner],[wEdge, wCenter, wCenter, wEdge], [wEdge, wCenter, wCenter, wEdge], [wCorner, wEdge, wEdge, wCorner]], dtype=torch.float32)
            mask.resize_((1,1,4,4))
            rhs_2h = torch.nn.functional.conv2d(residue_h, mask, bias=None, stride=2, padding=(1,1))

            #rhs_2h = torch.nn.functional.interpolate(input=residue_h, scale_factor=0.5,
            #    mode='bilinear', align_corners=False)
            rhs_2h.resize_((len_2h))
            residue_h.resize_((len_h))
            #self.projectToZero(rhs_2h, nodeType_2h, gridWidth_2h)
            print("printing rhs_2h")
            print(rhs_2h)

            print(torch.sqrt(torch.max(rhs_2h*rhs_2h)))

            q_2h = torch.zeros((len_2h), dtype=torch.float32)

            dInverse_2h = torch.zeros((len_2h), dtype = torch.float32)
            dInverse_2h = dInverse_2h + (h_2h * h_2h)/(self.centralWeight)
            dInverse_2h[0] = h_2h * h_2h
            dInverse_2h[len_2h-1] = h_2h * h_2h

            residue_2h = torch.zeros((len_2h), dtype=torch.float32)

            #Set the boundary
            self.setBoundary(lattice_2h, nodeType_2h, gridWidth_2h, value=0)

            jb_2h = JacobiSolver(lattice_2h, nodeType_2h, gridWidth_2h, 0)
            jb_2h.dampedJacobi(rhs_2h, q_2h, dInverse_2h, residue_2h, 50, 1e-5)

            #Upsample and add to self.lattice
            lattice_2h = jb_2h.lattice
            lattice_2h = lattice_2h.resize_((1,1,gridWidth_2h, gridWidth_2h))
            upsample = torch.nn.modules.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            lattice_upsampled = upsample(lattice_2h)
            lattice_upsampled = lattice_upsampled.resize_((len_h))
            self.setBoundary(v=lattice_upsampled, nT=self.nodeType, gW=self.gridWidth, value=0)
            self.lattice = self.lattice - lattice_upsampled

            self.writeToFile(i)
            jb_h = JacobiSolver(self.lattice, self.nodeType, self.gridWidth, b=self.b)
            jb_h.dampedJacobi(rhs_h, q_h, dInverse_h, residue_h, 1, 1e-5)
            self.lattice = jb_h.lattice
            residue_h = jb_h.getResidual(rhs_h)
            #self.writeToFile(i)
            #residue_h = self.smoother(residue_h, self.gridWidth, self.h, self.b)
            #self.lattice = jb_h.lattice

        #print(torch.sqrt(torch.max(residue_h*residue_h)))
        print(i)

    def resetBoundary(self, v):
        v.resize_((self.gridWidth, self.gridWidth))
        v[0,:] = -self.b/(self.h * self.h)
        v[self.gridWidth-1,:] = -self.b/(self.h * self.h)
        v[:,0] = -self.b/(self.h * self.h)
        v[:,self.gridWidth-1] = -self.b/(self.h * self.h)
        v.resize_((self.gridWidth*self.gridWidth))

    def setBoundary(self, v, nT, gW, value):
        v.resize_((gW, gW))
        v[0,:] = value
        v[gW-1,:] = value
        v[:,0] = value
        v[:,gW-1] = value
        v.resize_((gW*gW))

    # def setBoundary(self):
    #     self.lattice.resize_((self.gridWidth, self.gridWidth))
    #     self.lattice[0,:] = -self.b/(self.h * self.h)
    #     self.lattice[self.gridWidth-1,:] = -self.b/(self.h * self.h)
    #     self.lattice[:,0] = -self.b/(self.h * self.h)
    #     self.lattice[:,self.gridWidth-1] = -self.b/(self.h * self.h)
    #     self.lattice.resize_((self.gridWidth*self.gridWidth, 1))

def helloWorld():
    print("hello world")

def latticeShape(nodeType, gridWidth):
    nodeType[0,:] = 1.0
    nodeType[gridWidth-1,:] = 1.0
    nodeType[:,0] = 1.0
    nodeType[:,gridWidth-1] = 1.0

if __name__ == "__main__":
    helloWorld()
    gridWidth = 10
    lattice = -2.0 * torch.rand((gridWidth, gridWidth), dtype=torch.float32) + 1.0
    nodeType = torch.zeros((gridWidth, gridWidth))
    latticeShape(nodeType, gridWidth)
    print(nodeType)
    solver = LaplacianSolver(lattice, nodeType, gridWidth)
    solver.sampleFunction()
    solver.jacobiSolver()
