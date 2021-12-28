archive = xsmahe01.tar.gz

.PHONY: all pack clean

all:

pack:
	tar --ignore-failed-read -czvf $(archive) src/*.py audio/4cos.wav audio/clean_*.wav audio/bonus.wav

clean:
	rm $(archive)
