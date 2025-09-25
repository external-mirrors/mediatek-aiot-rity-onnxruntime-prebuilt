LIBDIR ?= "/usr/lib"
DATADIR ?= "/usr/share"
INCLUDEDIR ?= "/usr/include"
PWD ?= ""
LIBS=/usr/lib/*

all:

install:
	install -d $(INCLUDEDIR)
	cp -a ${PWD}/usr/include/* $(INCLUDEDIR)

	install -d $(LIBDIR)
	cp -a $(PWD)/usr/lib/* $(LIBDIR)

	install -d $(DATADIR)
	install -d $(DATADIR)/onnxruntime_example
	install $(PWD)/usr/share/onnxruntime_example/* $(DATADIR)/onnxruntime_example
