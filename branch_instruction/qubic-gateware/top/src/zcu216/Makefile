TGT=psbd
GENSRC=gensrc

all: $(TGT).bit

COMMITNUM = $(shell git log --pretty=oneline --abbrev-commit --abbrev=8|head -1 | awk '{print $$1}')
COMMITMSG = $(shell git log --pretty=oneline --abbrev-commit --abbrev=8|head -1 | awk '{print $$2}')
DATETIME = $(shell date  +'%Y%m%d%H%M%S')
AUTOCOMMITMSG=Makefileautomatedcommit
TOOLS="../../../submodules/tools/"
psbd: plps $(GENSRC)/bramwithctrl.tcl
	time vivado -mode batch -source ../../src/psbd.tcl
plps: clean pre gitrevision.v
	time vivado -mode batch -source ../../src/plps.tcl
pre: makefile.pre	
	make -f makefile.pre

gitcommit:
	python $(TOOLS)/gitauto.py -action autocommit
gitlist:
	python $(TOOLS)/gitauto.py -action list
gitdiff:
	python $(TOOLS)/gitauto.py -action diff
gitlog:
	git reflog --pretty=format:"%h %cd %cn %s"
gitrevision.v: gitcommit
	echo $(COMMITNUM)
	python $(TOOLS)/gitauto.py -action revisionverilog
	cat $(GENSRC)/gitrevision.v
gitpush:
	python submodules/tools/gitauto.py -action push


$(TGT).bit: $(TGT)
	$(eval FNAME:=$(TGT)_$(DATETIME)_$(COMMITNUM))
	echo $(FNAME)
	if [ ! -d "./bits" ]; then \
		mkdir bits; \
	fi
	cp -f vivado_project/$(TGT)/$(TGT).runs/impl_1/$(TGT).bit  ./bits/$(FNAME).bit
	cp -f vivado_project/$(TGT)/$(TGT).xsa ./bits/$(FNAME).xsa
	ln -sf ./bits/$(FNAME).bit $(TGT).bit
	ln -sf ./bits/$(FNAME).xsa $(TGT).xsa
	ls bits/$(FNAME).bit >> bits/bits
	echo $(FNAME)
	- cp -f vivado_project/$(TGT)/$(TGT).runs/impl_1/$(TGT).ltx ./bits/$(FNAME).ltx
	- ln -sf ./bits/$(FNAME).ltx $(TGT).ltx
	- cp -f vivado_project/$(TGT)/$(TGT).runs/impl_1/$(TGT).bin    ./bits/$(FNAME).bin
	printf "\a\a\a"

release: ./vivado_project/$(TGT)/$(TGT).xsa
	$(eval FNAME:=$(TGT)_$(DATETIME)_$(COMMITNUM))
	$(eval BUILDDIR:=$(shell pwd))
	mkdir $(COMMITNUM)
	cp -f vivado_project/$(TGT)/$(TGT).xsa ./$(COMMITNUM)/$(FNAME).xsa
	cp ./gensrc/channel_config.json ./$(COMMITNUM)/
	cp ./gensrc/channel_map.txt ./$(COMMITNUM)/
	cp ./gensrc/bram.json ./$(COMMITNUM)/
	cp ./cfgregs.json ./$(COMMITNUM)/
	cp ./dspregs.json ./$(COMMITNUM)/
	cp ./rfdc.json ./$(COMMITNUM)/
	tar -czvf $(COMMITNUM).tar.gz $(COMMITNUM)
	cd ..
	tar -czvf $(BUILDDIR).tar.gz $(BUILDDIR)

clean:
	rm -rf vivado_project
