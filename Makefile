classpath = .:..:share/OpenCV/java/opencv-248.jar:lib/commons-math3-3.2.jar
copts = -Xlint -Xfatal-warnings -deprecation -optimise
libpath = share/OpenCV/java
ropts = -cp $(classpath) -Djava.library.path=$(libpath) -J-Xmx8G
scalac = fsc $(copts)

export scalac

.PHONY: all clean cvtools detector go prob run test tests train util

all: util prob detector cvtools train test tests

clean:
	@-$(MAKE) clean -C cvtools -s
	@-$(MAKE) clean -C detector -s
	@-$(MAKE) clean -C prob -s
	@-$(MAKE) clean -C test -s
	@-$(MAKE) clean -C tests -s
	@-$(MAKE) clean -C train -s
	@-$(MAKE) clean -C util -s

cvtools: util
	@-$(MAKE) cvtools -C cvtools --no-print-directory

detector: util prob
	@-$(MAKE) detector -C detector --no-print-directory

go: tests run-tests

prob:
	@-$(MAKE) prob -C prob --no-print-directory

# to pass args do 'make run-test args="arg0 arg1..."'
run-test: test
	scala $(ropts) imgdetect.test.TestBayesSuper $(args)

run-tests: tests
	scala $(ropts) imgdetect.tests.CVTest

# to pass args do 'make run-train args="arg0 arg1..."'
run-train: train
	scala $(ropts) imgdetect.train.TrainBayesSuper $(args)

test: util detector cvtools
	@-$(MAKE) test -C test --no-print-directory

tests: cvtools
	@-$(MAKE) tests -C tests --no-print-directory

train: util cvtools
	@-$(MAKE) train -C train --no-print-directory

util:
	@-$(MAKE) util -C util --no-print-directory