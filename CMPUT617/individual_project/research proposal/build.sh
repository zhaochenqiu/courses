# 用pdfLaTeX编译你的 .tex 文件 , 这是生成一个 .aux 的文件, 这告诉 BibTeX 将使用那些应用.
pdflatex research.tex 

# 用BibTeX 生成 .bib 文件.
bibtex research.aux

# 再次用pdfLaTeX 编译你的 .tex 文件, 这个时候在文档中已经包含了参考文献, 但此时引用的编号可能不正确
pdflatex research.tex

# 最后用 再再次pdfLaTeX 编译你的 .tex 文件, 如果一切顺利的话, 这是所有东西都已正常了
pdflatex research.tex

# 打开pdf文件
evince research.pdf
