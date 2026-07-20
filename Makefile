.PHONY: papers neurips video4real clean-papers clean-neurips clean-video4real

papers: neurips video4real

neurips:
	cd paper && latexmk -pdf neurips.tex

video4real:
	cd paper && TEXINPUTS=./eccv2026//: BIBINPUTS=.: latexmk -pdf -bibtex video4real.tex

clean-papers: clean-neurips clean-video4real

clean-neurips:
	cd paper && latexmk -C neurips.tex

clean-video4real:
	cd paper && latexmk -C video4real.tex
