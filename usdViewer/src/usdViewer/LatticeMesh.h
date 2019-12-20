#include "AnimatedMesh.h"

#include <Eigen/Dense>

#include <map>

template<class T>
struct LatticeMesh : public AnimatedMesh<T>
{
    using Base = AnimatedMesh<float>;

    // from AnimatedTetrahedonMesh
    using Base::m_meshElements;
    using Base::m_particleX;
    using Base::initializeUSD;
    using Base::initializeTopology;
    using Base::initializeParticles;
    using Vector3 = typename Base::Vector3;
    using Base::writeFrame;

    std::array<int, 3> m_cellSize; // dimensions in grid cells
    int m_radius; // radius of sphere in grid cells
    T m_gridDX;

    std::vector<std::array<int, 3>> m_activeCells; // Marks the "active" cells in the lattice
    std::map<std::array<int, 3>, int> m_activeNodes; // Maps the "active" nodes to their particle index
    
    void setLatticeValues(std::vector<std::vector<T>> inputLattice) {
        int particlePos = 0;
        for(auto& p: m_particleX) {
            std::cout << p << std::endl;
            p[0] = inputLattice[particlePos][0];
            p[1] = inputLattice[particlePos][1];
            p[2] = inputLattice[particlePos][2];
            std::cout << p << std::endl;
            particlePos += 1;
        }
        
    }
    
    void initialize(std::vector<std::vector<T>> inputLattice, std::string nameOfUSDFile)
    {
        initializeUSD("Demo3D.usda");

        // Activate cells within a sphere of radius m_radius (in cells)

        for(int cell_i = 0; cell_i < m_cellSize[0]; cell_i++)
        for(int cell_j = 0; cell_j < m_cellSize[1]; cell_j++)
        for(int cell_k = 0; cell_k < m_cellSize[2]; cell_k++){
            m_activeCells.push_back(std::array<int, 3>{cell_i, cell_j, cell_k});

        }

        std::cout << "Created a model including " << m_activeCells.size() << " lattice cells" <<std::endl;

        // Create (uniquely numbered) particles at the node corners of active cells
        int particlePos = 0;
        for(const auto& cell: m_activeCells){
            std::array<int, 3> node;
            for(node[0] = cell[0]; node[0] <= cell[0]+1; node[0]++)
            for(node[1] = cell[1]; node[1] <= cell[1]+1; node[1]++)
            for(node[2] = cell[2]; node[2] <= cell[2]+1; node[2]++){
                auto search = m_activeNodes.find(node);
                if(search == m_activeNodes.end()){ // Particle not yet created at this lattice node location -> make one
                    m_activeNodes.insert({node, m_particleX.size()});
//                    m_particleX.emplace_back(m_gridDX * T(node[0]), m_gridDX * T(node[1]), m_gridDX * T(node[2]));
                    //int pos = gridToParticleID(node[0], node[1], node[2]);
                    m_particleX.emplace_back(inputLattice[particlePos][0],
                                             inputLattice[particlePos][1],
                                             inputLattice[particlePos][2]);
                    particlePos += 1;
                }
            }
        }
        std::cout << "Model contains " << m_particleX.size() << " particles" << std::endl;

        // Make tetrahedra out of all active cells (6 tetrahedra per cell)

        for(const auto& cell: m_activeCells){
            std::cout << cell[0] << " " << cell[1] << " "<< cell[2] << std::endl;
            int vertexIndices[2][2][2];
            for(int i = 0; i <= 1; i++)
            for(int j = 0; j <= 1; j++)
            for(int k = 0; k <= 1; k++){
                std::array<int, 3> node{cell[0] + i, cell[1] + j, cell[2] + k};
                auto search = m_activeNodes.find(node);
                if(search != m_activeNodes.end())
                    vertexIndices[i][j][k] = search->second;
                else
                    throw std::logic_error("particle at cell vertex not found");
            }
            std::cout << vertexIndices[0][0][0] << std::endl;
            std::cout << vertexIndices[1][1][1] << std::endl;
            m_meshElements.push_back(std::array<int, 4>{ vertexIndices[0][0][0], vertexIndices[1][0][0], vertexIndices[1][1][0], vertexIndices[1][1][1]});
            m_meshElements.push_back(std::array<int, 4>{ vertexIndices[0][0][0], vertexIndices[1][0][0], vertexIndices[1][1][1], vertexIndices[1][0][1]});
            m_meshElements.push_back(std::array<int, 4>{ vertexIndices[0][0][0], vertexIndices[1][0][1], vertexIndices[1][1][1], vertexIndices[0][0][1]});
            m_meshElements.push_back(std::array<int, 4>{ vertexIndices[0][0][0], vertexIndices[1][1][1], vertexIndices[0][1][1], vertexIndices[0][0][1]});
            m_meshElements.push_back(std::array<int, 4>{ vertexIndices[0][0][0], vertexIndices[1][1][1], vertexIndices[0][1][0], vertexIndices[0][1][1]});
            m_meshElements.push_back(std::array<int, 4>{ vertexIndices[0][0][0], vertexIndices[1][1][0], vertexIndices[0][1][0], vertexIndices[1][1][1]});
        }
        
        // Perform the USD-specific initialization of topology & particles
        // (this will also create a boundary *surface* to visualuze

        initializeTopology();
        initializeParticles();

        // Check particle indexing in mesh

        for(const auto& element: m_meshElements)
            for(const auto vertex: element)
                if(vertex < 0 || vertex >= m_particleX.size())
                    throw std::logic_error("mismatch between mesh vertex and particle array");
    }
    
//    void setFrame() {
//        for(int node_i = 1; node_i <= m_cellSize[0]; node_i++) {
//            for(int node_j = 1; node_j <= m_cellSize[1]; node_j++) {
//                for(int node_k = 1; node_k <= m_cellSize[2]; node_k++) {
//
//                }
//            }
//        }
//    }
    
    void regFrame(std::vector<std::vector<T>> inputLattice) {
        static int frameNo = 1;
        setLatticeValues(inputLattice);
//        setFrame();
        writeFrame(frameNo);
        frameNo += 1;
    }
private:
    int gridToParticleID(int i, int j, int k) {
        return (i * (m_cellSize[1]+1) * (m_cellSize[2]+1))+(j * (m_cellSize[2]+1))+k;
    }
    
};
