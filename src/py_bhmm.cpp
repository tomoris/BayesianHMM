
#include "bayesian_hmm.hpp"
#include "data_container.hpp"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<std::vector<MyTagIdType>> BayesianHMM::TrainForPython(const std::vector<std::vector<MyWordIdType>> &corpus, std::vector<std::vector<MyTagIdType>> &tag_corpus, const int epoch)
{
    this->Train(corpus, tag_corpus, epoch);
    return tag_corpus;
}

PYBIND11_PLUGIN(py_bhmm)
{
    py::module m("py_bhmm", "mylibs made by pybind11");
    py::class_<BayesianHMM>(m, "BayesianHMM")
        .def(py::init<int, int, float, float>())
        .def("Train", &BayesianHMM::TrainForPython);
    py::class_<DataContainer>(m, "DataContainer")
        .def(py::init<std::string, std::string>())
        .def_readwrite("corpus", &DataContainer::corpus)
        .def_readwrite("tag_corpus", &DataContainer::tag_corpus)
        .def("GetWordVocabSize", &DataContainer::GetWordVocabSize);

    return m.ptr();
}
