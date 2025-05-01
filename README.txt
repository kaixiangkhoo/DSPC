Please include the "opencv_world4110d.dlll" file into this folder or else the program cannot run.

CONFIGURATION
--------------
1. C/C++ -> Additional Include Directories -> "yourpath\include";"yourpath\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include"
2. CUDA C/C++ -> Command Line -> Paste this "-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
3. Linker -> General -> Additional Library Directories -> "yourpath\x64\vc16\lib";"yourpath\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" 
4. Linker -> Input -> Add "opencv_world4110d.lib;"


PYTHON SCRIPT
--------------
1. python script.py --run --base-path imageDataset
2. python script.py --interactive (Displays cpp CLI UI)
2. python script.py --visualize "name.csv" --output-dir "outputFileName"
(Must first train model, recommended to run option #2)