call "D:\Program Files\latex\tools_2014.bat" 
rem call "E:\Program Files\latex\tools_2014.bat" 
rem set path=%path%;C:\texlive\2017\bin\win32;
call "C:\latex\tools_2017.bat"

pdflatex research.tex 
bibtex research.aux 
pdflatex research.tex 
pdflatex research.tex
run.bat
