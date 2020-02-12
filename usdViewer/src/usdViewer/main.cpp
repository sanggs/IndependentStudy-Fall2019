
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include "LatticeMesh.h"

using namespace std;

void readPoints(int size, vector< vector<float> > &points)
{
    fstream fin;
    fin.open("../../../3DPoints.csv", ios::in);
    
    int count = 0;
    while (fin.good()) {
        string line;
        getline(fin, line);
        
        stringstream s(line);

        std::string word;
        while (getline(s, word, ',')) {
            points[count].push_back(stod(word));
        }
        count++;
    }
}

int main(int argc, char *argv[]) {
    std::cout << "abcd" << std::endl;
    int numLines = stoi(argv[1]);
    vector< std::vector<float> > temp(numLines);
    readPoints(numLines, temp);
    
    //Populate points in USD
    //const int latticeSize = 10;
    LatticeMesh<float> simulationMesh;
    simulationMesh.m_cellSize = { 5, 4, 3};
    simulationMesh.m_gridDX = 0.5;
    
    int totalFrames = numLines/((simulationMesh.m_cellSize[0]+1) *
                                (simulationMesh.m_cellSize[1]+1) *
                                (simulationMesh.m_cellSize[2]+1));
    
    int startPos = 0;
    int len = ((simulationMesh.m_cellSize[0]+1) *
               (simulationMesh.m_cellSize[1]+1) *
               (simulationMesh.m_cellSize[2]+1));
    for(int t = 0; t < totalFrames; t++) {
        std::vector<std::vector<float>> gridPos;
        std::vector<bool> visited(len, false);
        
        for(int i = startPos; i < startPos+len; i++) {
            gridPos.push_back(temp[i]);
        }
        
        if ((gridPos.size()) == 0) {
            exit(0);
        }
        
        if(t == 0) {
            simulationMesh.initialize(gridPos, "result.usda");
            std::cout << "Step 0 done" << std::endl;
            simulationMesh.writeFrame(t);
        }
        else {
            simulationMesh.regFrame(gridPos);
        }
        
        startPos += ((simulationMesh.m_cellSize[0]+1) *
                     (simulationMesh.m_cellSize[1]+1) *
                     (simulationMesh.m_cellSize[2]+1));
    }
    
    simulationMesh.writeUSD();
    return 0;
}

