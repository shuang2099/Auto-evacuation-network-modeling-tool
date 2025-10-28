# Auto-evacuation-network-modeling-tool
A standalone software tool designed to automatically generate evacuation topological networks from floor plan images and perform evacuation simulations with visualization

# 1. Introduction
Auto-evacuation-network-modeling-tool is a standalone software tool designed to automatically generate evacuation topological networks from floor plan images and perform evacuation simulations with visualization. The tool processes input floor plans through a series of steps including semantic segmentation, morphological processing, path analysis, network modeling, and simulation.

Note:​​ This is a pre-packaged executable version - ​no Python installation required!​​

# 2. System Requirements
Operating System:​​ Windows 10/11 (64-bit)  
Memory:​​ Minimum 2GB RAM (4GB or more recommended)  
​Disk Space:​​ At least 500MB free space  
​Dependencies:​​ FFmpeg (required for evacuation animation generation)  

# 3. FFmpeg Installation (Required for Visualization)
Click to download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z  
Add the path to the bin folder (e.g., C:\ffmpeg\bin) to your system's PATH environment variable.  
Restart your command prompt or computer, then open a new command prompt and type ffmpeg -version to verify the installation.

# 4. Quick Start Guide
1. Download the Application
   https://github.com/shuang2099/Auto-evacuation-network-modeling-tool/releases/download/evac_tool.exe/evac_tool.exe
3. Create New Project  
   Click the ​​"Choose path"​​ button to select a working directory.  
   Click the ​​"Add"​​ button to import floor plan images (PNG and JPG formats supported).  
3. Modeling Process (Execute in Sequential Order)  
   ​Pick Stair Nodes: Mark staircase positions on each floor plan.  
​   Run Recognition: Automatically identify rooms and doors.  
​   Run Results refinement: Refine the semantic segmentation results.  
​   Run Routes analysis: Analyze evacuation paths.  
   Run Network modeling: Create the final topological network model.  
4. Simulation & Visualizationc  
   ​Input Stair Parameters: Set staircase-related parameters.  
​   Input Ratio: Use the ratio input tool.  
​   Build table: Prepare the Excel input table for simulation.  
​   Run Macro Simulation: Execute evacuation calculations.  
​   Run Visualization: Generate time graphs and evacuation animation.  
5. Export Results  
   Click ​​"PDF report"​​ to generate a comprehensive PDF report containing all results.
