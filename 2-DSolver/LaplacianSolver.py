import torch
import numpy as np
import math
import numba
from JacobiSolver import JacobiSolver
from ConjugateGradientSolver import CGSolver

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
        l = self.lattice.view((self.gridWidth * self.gridWidth))
        if i == 0:
            f = open("points1.csv", "w")
        else:
            f = open("points1.csv", "a")
        npArray = l.numpy()
        print(npArray.shape)
        np.savetxt(f, npArray[None], delimiter=',')
        f.close()

    def projectToZero(self, v, nodeType, gW):
        v.resize_((gW, gW))
        nodeType.resize_((gW, gW))
        v = v * nodeType
        nodeType.resize_((gW*gW))
        return v.resize_((gW*gW))

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
        rhs_h = self.projectToZero(rhs_h, self.nodeType, self.gridWidth)
        # print(rhs_h.shape)
        # return

        q_h = torch.zeros((len_h), dtype=torch.float32)

        #Set the boundary
        self.lattice = self.setBoundary(v=self.lattice, nT=self.nodeType, gW=self.gridWidth, value=0)
        #self.lattice.resize_((len_h))

        dInverse_h = torch.zeros((len_h), dtype = torch.float32)
        dInverse_h = dInverse_h + (self.h * self.h)/(self.centralWeight)
        dInverse_h[0] = self.h * self.h
        dInverse_h[len_h-1] = self.h * self.h
        print("printing self.lattice")
        print(self.lattice)
        self.writeToFile(0)
        residue_h = torch.zeros((len_h), dtype=torch.float32)

        #self.lattice = jb_h.lattice
        # residue_h = jb_h.getResidual(rhs_h)
        # print(torch.sqrt(torch.max(residue_h*residue_h)))
        for i in range(1, 2, 1):
            #Do it for 2h grid and then call solve
            len_2h = int(self.gridWidth/2 * self.gridWidth/2)
            gridWidth_2h = int(self.gridWidth/2)
            h_2h = gridWidth_2h - 1;

            jb_h = JacobiSolver(lattice=self.lattice, nodeType=self.nodeType, gridWidth=self.gridWidth, b=self.b)
            jb_h.dampedJacobi(rhs_h, q_h, dInverse_h, residue_h, 1, 1e-5)
            self.lattice = jb_h.lattice
            residue_h = jb_h.getResidual(rhs=rhs_h)

            self.writeToFile(i)
            #residue_h = self.smoother(residue_h, self.gridWidth, self.h, self.b)
            print("printing residue_h")
            print(residue_h)

            lattice_2h = torch.zeros((len_2h), dtype=torch.float32)

            #nodeType
            nodeType_h = self.nodeType.resize_((1,1,self.gridWidth, self.gridWidth))
            maxpool = torch.nn.MaxPool2d((2,2), stride=2)
            nodeType_2h = maxpool(nodeType_h)
            print(nodeType_2h)
            nodeType_2h.resize_((len_2h))

            #rhs => residue
            wCorner = 1/16
            wEdge = 3/16
            wCenter = 9/16
            print("residue_h")
            #print(residue_h.view((self.gridWidth, self.gridWidth)))
            residue_h = residue_h.resize_((1,1,self.gridWidth, self.gridWidth))
            #residue_h = -1 * residue_h
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
            lattice_2h = self.setBoundary(lattice_2h, nodeType_2h, gridWidth_2h, value=0)

            jb_2h = JacobiSolver(lattice_2h, nodeType_2h, gridWidth_2h, 0)
            jb_2h.dampedJacobi(rhs_2h, q_2h, dInverse_2h, residue_2h, 100, 1e-5)

            #Upsample and add to self.lattice
            lattice_2h = jb_2h.lattice
            lattice_2h = lattice_2h.resize_((1,1,gridWidth_2h, gridWidth_2h))
            upsample = torch.nn.modules.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            lattice_upsampled = upsample(lattice_2h)
            lattice_upsampled = lattice_upsampled.resize_((len_h))
            #lattice_upsampled = self.setBoundary(v=lattice_upsampled, nT=self.nodeType, gW=self.gridWidth, value=0)
            print(lattice_upsampled)
            self.lattice = self.lattice + lattice_upsampled

            self.writeToFile(i)
            jb_h = JacobiSolver(self.lattice, self.nodeType, self.gridWidth, b=self.b)
            #jb_h.dampedJacobi(rhs_h, q_h, dInverse_h, residue_h, 1, 1e-5)
            #self.lattice = jb_h.lattice
            residue_h = jb_h.getResidual(rhs_h)
            self.writeToFile(i)
            #residue_h = self.smoother(residue_h, self.gridWidth, self.h, self.b)
            #self.lattice = jb_h.lattice

        #print(torch.sqrt(torch.max(residue_h*residue_h)))
        print(i)

    def CGSolve(self, lattice, nodeType, gW, h, rhs_h, iterations, residue):
        len_h = gW * gW
        q_h = torch.zeros((len_h), dtype = torch.float32)
        s_h = torch.zeros((len_h), dtype = torch.float32)
        residue_h = torch.zeros((len_h), dtype = torch.float32)
        cg_h = CGSolver(lattice, nodeType, gW, self.b, h, rhs_h, q_h, s_h, residue_h, iterations, residue)
        cg_h.solve()
        lattice = cg_h.getSolution()
        residue_h = cg_h.getResidual()
        return lattice, residue_h

    def restrict(self, v, gW, gW_2h):
        # wCorner = 1/16
        # wEdge = 3/16
        # wCenter = 9/16
        v.resize_((1, 1, gW, gW))
        # mask = torch.tensor([[wCorner, wEdge, wEdge, wCorner],[wEdge, wCenter, wCenter, wEdge], [wEdge, wCenter, wCenter, wEdge], [wCorner, wEdge, wEdge, wCorner]], dtype=torch.float32)
        # mask.resize_((1,1,4,4))
        # v_2h = torch.nn.functional.conv2d(v, mask, bias=None, stride=2, padding=(1,1))
        v_2h = torch.nn.functional.interpolate(v, scale_factor = 0.5, mode = 'bilinear', align_corners=True)
        return v_2h.resize_((gW_2h*gW_2h))

    def interpolate(self, v_2h, gW, gW_2h):
        v_2h.resize_((1, 1, gW_2h, gW_2h))
        upsample = torch.nn.modules.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        v_h = upsample(v_2h)
        return v_h.resize_((gW*gW))

    def multigridCGSolve(self):
        self.writeToFile(0)
        for i in range(1, 61):
            rhs_h = torch.zeros((self.gridWidth*self.gridWidth), dtype = torch.float32)
            rhs_h = rhs_h + (-self.b/(self.h*self.h))
            self.lattice, residue_h = self.CGSolve(self.lattice, self.nodeType, self.gridWidth, self.h, rhs_h, 0, 1e-5)
            self.writeToFile(1)

            rhs_2h = self.restrict(residue_h, self.gridWidth, int(self.gridWidth/2))

            lattice_2h = torch.zeros((int(self.gridWidth/2)*int(self.gridWidth/2)), dtype=torch.float32)

            maskNode = 1.0 - self.nodeType
            maskNode.resize_((1,1,self.gridWidth, self.gridWidth))
            maxpool = torch.nn.MaxPool2d((2,2), stride=2)
            nodeType_2h = maxpool(maskNode)
            nodeType_2h.resize_((int(self.gridWidth/2), int(self.gridWidth/2)))
            nodeType_2h = 1.0 - nodeType_2h
            maskNode.resize_((self.gridWidth, self.gridWidth))

            lattice_2h, residue_2h = self.CGSolve(lattice_2h, nodeType_2h, int(self.gridWidth/2), int(self.gridWidth/2)-1, rhs_2h, 1000, 1e-18)
            error_h = self.interpolate(lattice_2h, self.gridWidth, int(self.gridWidth/2))
            error_h = self.setBoundary(error_h, self.nodeType, self.gridWidth, 0.0)
            self.lattice = self.lattice + error_h
            self.writeToFile(i)

            self.lattice, residue_h = self.CGSolve(self.lattice, self.nodeType, self.gridWidth, self.h, rhs_h, 0, 1e-5)
            self.writeToFile(3)

    def setBoundary(self, v, nT, gW, value):
        v.resize_((gW, gW))
        nT.resize_((gW, gW))
        val = (1.0-nT)*value
        v = v * nT + val
        nT.resize_((gW*gW))
        return v.resize_((gW*gW))

def helloWorld():
    print("hello world")

def latticeShape(nodeType, gridWidth, boundary):
    nodeType += 1.0
    nodeType[0:boundary,:] = 0.0
    nodeType[gridWidth-boundary:gridWidth,:] = 0.0
    nodeType[:,0:boundary] = 0.0
    nodeType[:,gridWidth-boundary:gridWidth] = 0.0

def populateLatticeValues(lattice, gridWidth, boundary):
    aoi = gridWidth-(2*boundary)
    noise = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.5]))
    lowLevelNoise = noise.sample((int(aoi/8),int(aoi/8))).resize_((int(aoi/8),int(aoi/8)))
    lowLevelNoise[0,:] = 0.0
    lowLevelNoise[int(aoi/8)-1,:] = 0.0
    lowLevelNoise[:,0] = 0.0
    lowLevelNoise[:,int(aoi/8)-1] = 0.0
    lowLevelNoise.resize_((1, 1, int(aoi/8),int(aoi/8)))
    upsample = torch.nn.modules.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
    highLevelNoise = upsample(lowLevelNoise)
    highLevelNoise.resize_((aoi, aoi))

    #lattice[boundary:gridWidth-boundary, boundary:gridWidth-boundary] = noise.sample((aoi,aoi)).resize_((aoi, aoi))
    lattice[boundary:gridWidth-boundary, boundary:gridWidth-boundary] = highLevelNoise

if __name__ == "__main__":
    helloWorld()
    gridWidth = 100
    boundary = 10
    #lattice = torch.rand((gridWidth, gridWidth), dtype=torch.float32)
    lattice = torch.zeros((gridWidth, gridWidth), dtype=torch.float32)
    populateLatticeValues(lattice, gridWidth, boundary)
    nodeType = torch.zeros((gridWidth, gridWidth))
    latticeShape(nodeType, gridWidth, boundary)
    print(nodeType)

    #print(nodeType)
    #b = torch.from_numpy(a)
    solver = LaplacianSolver(lattice, nodeType, gridWidth)
    solver.sampleFunction()
    #solver.jacobiSolver()
    solver.multigridCGSolve()
