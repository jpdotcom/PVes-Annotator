AnnotateGui Executable

1. Prerequisites for Deployment 

Verify that MATLAB Runtime(R2024a) is installed.   
If not, you can run the MATLAB Runtime installer.
To find its location, enter
  
    >>mcrinstaller
      
at the MATLAB prompt.
NOTE: You will need administrator rights to run the MATLAB Runtime installer. 

Alternatively, download and install the Macintosh version of the MATLAB Runtime for R2024a 
from the following link on the MathWorks website:

    https://www.mathworks.com/products/compiler/mcr/index.html
   
For more information about the MATLAB Runtime and the MATLAB Runtime installer, see 
"Distribute Applications" in the MATLAB Compiler documentation  
in the MathWorks Documentation Center.

2. Files to Deploy and Package

Files to Package for Standalone 
================================
-run_AnnotateGui.sh (shell script for temporarily setting environment variables and 
                     executing the application)
   -to run the shell script, type
   
       ./run_AnnotateGui.sh <mcr_directory> <argument_list>
       
    at Linux or Mac command prompt. <mcr_directory> is the directory 
    where MATLAB Runtime(R2024a) is installed or the directory where 
    MATLAB is installed on the machine. <argument_list> is all the 
    arguments you want to pass to your application. For example, 

    If you have MATLAB Runtime(R2024a) installed in 
    /mathworks/home/application/R2024a, run the shell script as:
    
       ./run_AnnotateGui.sh /mathworks/home/application/R2024a
       
    If you have MATLAB installed in /mathworks/devel/application/matlab, 
    run the shell script as:
    
       ./run_AnnotateGui.sh /mathworks/devel/application/matlab
-MCRInstaller.zip 
    Note: if end users are unable to download the MATLAB Runtime using the
    instructions in the previous section, include it when building your 
    component by clicking the "Runtime included in package" link in the
    Deployment Tool.
-The Macintosh bundle directory structure AnnotateGui.app 
    Note: this can be stored in an archive file with the zip command 
    zip -r AnnotateGui.zip AnnotateGui.app
    or the tar command 
    tar -cvf AnnotateGui.tar AnnotateGui.app
-This readme file 



3. Definitions

For information on deployment terminology, go to
https://www.mathworks.com/help and select MATLAB Compiler >
Getting Started > About Application Deployment >
Deployment Product Terms in the MathWorks Documentation
Center.

4. Appendix 

A. Mac systems:
In the following directions, replace MR/R2024a by the directory on the target machine 
   where MATLAB is installed, or MR by the directory where the MATLAB Runtime is 
   installed.

If the environment variable DYLD_LIBRARY_PATH is undefined, set it to the following 
   string:

MR/R2024a/runtime/maca64:MR/R2024a/sys/os/maca64:MR/R2024a/bin/maca64

If it is defined, set it to the following:

${DYLD_LIBRARY_PATH}:MR/R2024a/runtime/maca64:MR/R2024a/sys/os/maca64:MR/R2024a/bin/maca64

    For more detailed information about setting the MATLAB Runtime paths, see Package and 
   Distribute in the MATLAB Compiler documentation in the MathWorks Documentation Center.


     
        NOTE: To make these changes persistent after logout on Linux 
              or Mac machines, modify the .cshrc file to include this  
              setenv command.
        NOTE: The environment variable syntax utilizes forward 
              slashes (/), delimited by colons (:).  
        NOTE: When deploying standalone applications, you can
              run the shell script file run_AnnotateGui.sh 
              instead of setting environment variables. See 
              section 2 "Files to Deploy and Package".    



5. Launching application using Macintosh finder

If the application is purely graphical, that is, it doesn't read from standard in or 
write to standard out or standard error, it may be launched in the finder just like any 
other Macintosh application.



