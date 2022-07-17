"""!

@brief General definitions of samples.

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2020
@copyright BSD-3-Clause

"""

import pyclustering.samples as samples
import os


## Path to samples module.
DEFAULT_SAMPLE_PATH = samples.__path__[0] + os.sep + "samples" + os.sep


class SIMPLE_SAMPLES:
    """!
    @brief The Simple Suite offers a variety of simple clustering problems.
    @details The samples are supposed to use for unit-testing and common algorithm abilities to found out
              run-time problems.
    
    """

    ## Simple Sample collection path.
    COLLECTION_PATH = DEFAULT_SAMPLE_PATH + "simple" + os.sep

    SAMPLE_SIMPLE1 = COLLECTION_PATH + "Simple01.data"
    SAMPLE_SIMPLE2 = COLLECTION_PATH + "Simple02.data"
    SAMPLE_SIMPLE3 = COLLECTION_PATH + "Simple03.data"
    SAMPLE_SIMPLE4 = COLLECTION_PATH + "Simple04.data"
    SAMPLE_SIMPLE5 = COLLECTION_PATH + "Simple05.data"
    SAMPLE_SIMPLE6 = COLLECTION_PATH + "Simple06.data"
    SAMPLE_SIMPLE7 = COLLECTION_PATH + "Simple07.data"
    SAMPLE_SIMPLE8 = COLLECTION_PATH + "Simple08.data"
    SAMPLE_SIMPLE9 = COLLECTION_PATH + "Simple09.data"
    SAMPLE_SIMPLE10 = COLLECTION_PATH + "Simple10.data"
    SAMPLE_SIMPLE11 = COLLECTION_PATH + "Simple11.data"
    SAMPLE_SIMPLE12 = COLLECTION_PATH + "Simple12.data"
    SAMPLE_SIMPLE13 = COLLECTION_PATH + "Simple13.data"
    SAMPLE_SIMPLE14 = COLLECTION_PATH + "Simple14.data"
    SAMPLE_SIMPLE15 = COLLECTION_PATH + "Simple15.data"
    SAMPLE_ELONGATE = COLLECTION_PATH + "Elongate.data"


class SIMPLE_ANSWERS:
    """!
    @brief Proper clustering results of samples from 'SIMPLE_SAMPLES'.

    @see SIMPLE_SAMPLES

    """

    COLLECTION_PATH = DEFAULT_SAMPLE_PATH + "simple" + os.sep

    ANSWER_SIMPLE1 = COLLECTION_PATH + "Simple01.answer"
    ANSWER_SIMPLE2 = COLLECTION_PATH + "Simple02.answer"
    ANSWER_SIMPLE3 = COLLECTION_PATH + "Simple03.answer"
    ANSWER_SIMPLE4 = COLLECTION_PATH + "Simple04.answer"
    ANSWER_SIMPLE5 = COLLECTION_PATH + "Simple05.answer"
    ANSWER_SIMPLE6 = COLLECTION_PATH + "Simple06.answer"
    ANSWER_SIMPLE7 = COLLECTION_PATH + "Simple07.answer"
    ANSWER_SIMPLE8 = COLLECTION_PATH + "Simple08.answer"
    ANSWER_SIMPLE9 = COLLECTION_PATH + "Simple09.answer"
    ANSWER_SIMPLE10 = COLLECTION_PATH + "Simple10.answer"
    ANSWER_SIMPLE11 = COLLECTION_PATH + "Simple11.answer"
    ANSWER_SIMPLE12 = COLLECTION_PATH + "Simple12.answer"
    ANSWER_SIMPLE13 = COLLECTION_PATH + "Simple13.answer"
    ANSWER_SIMPLE14 = COLLECTION_PATH + "Simple14.answer"
    ANSWER_SIMPLE15 = COLLECTION_PATH + "Simple15.answer"
    ANSWER_ELONGATE = COLLECTION_PATH + "Elongate.answer"




