.PHONY: video4real clean-video4real

video4real:
	cd paper && TEXINPUTS=./eccv2026//: BIBINPUTS=.: latexmk -pdf -bibtex video4real.tex

clean-video4real:
	cd paper && latexmk -C video4real.tex
