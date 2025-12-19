LIBDIR ?= "/usr/lib"
DATADIR ?= "/usr/share"
INCLUDEDIR ?= "/usr/include"
PWD ?= ""
LIBS=/usr/lib/*
ORT_EP ?= "0"

all:

install:

ifeq ($(ORT_EP),1)
	install -d $(INCLUDEDIR)
	cp -a ${PWD}/ort_w_ep/usr/include/* $(INCLUDEDIR)

	install -d $(LIBDIR)
	cp -a $(PWD)/ort_w_ep/usr/lib/* $(LIBDIR)

	install -d $(DATADIR)
	install -d $(DATADIR)/onnxruntime_example
	install -d $(DATADIR)/onnxruntime_benchmark
	install $(PWD)/ort_w_ep/usr/share/onnxruntime_example/* $(DATADIR)/onnxruntime_example
	install $(PWD)/ort_w_ep/usr/share/onnxruntime_benchmark/* $(DATADIR)/onnxruntime_benchmark
else
	install -d $(INCLUDEDIR)
	cp -a ${PWD}/ort_wo_ep/usr/include/* $(INCLUDEDIR)

	install -d $(LIBDIR)
	cp -a $(PWD)/ort_wo_ep/usr/lib/* $(LIBDIR)
	
	install -d $(DATADIR)
	install -d $(DATADIR)/onnxruntime_example
	install $(PWD)/ort_wo_ep/usr/share/onnxruntime_example/* $(DATADIR)/onnxruntime_example
endif
