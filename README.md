# IndependentStudy-Fall2019
##Projective Dynamics Simulator
1. Command to run: python3 LatticeMesh.py
2. Particle positions after every frame are written into 3DPoints.csv
3. To generate Pixar-USD file from this 3DPoints.csv
    * cd usdViewer
    * cmake -G Xcode .
    * Open usdViewer in Xcode and run the project. Ensure 3DPoints.csv is present in the main folder. Add command line argument as number of lines in 3DPoints.csv file. USD file will be written into the Debug folder inside usdViewer/src/usdViewer/
    * To view the usda file, run usdview Demo3D.usda
## MultiGrid Solver
1. Inside the 2-DSolver folder. 
2. Command to run: python3 LaplacianSolver.py
3. Demos inside 2-DSolver/results
