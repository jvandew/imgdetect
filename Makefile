.PHONY: all clean cvtest cvtools go run test train util

all: util cvtools train test cvtest

clean:
	@-$(MAKE) clean -C cvtest -s
	@-$(MAKE) clean -C cvtools -s
	@-$(MAKE) clean -C test -s
	@-$(MAKE) clean -C train -s
	@-$(MAKE) clean -C util -s

cvtest: cvtools
	@-$(MAKE) cvtest -C cvtest --no-print-directory

cvtools: util
	@-$(MAKE) cvtools -C cvtools --no-print-directory

go: cvtest run

run:
	@-$(MAKE) run -C cvtest --no-print-directory

test:
	@-$(MAKE) test -C test --no-print-directory

train: util cvtools
	@-$(MAKE) train -C train --no-print-directory

util:
	@-$(MAKE) util -C util --no-print-directory