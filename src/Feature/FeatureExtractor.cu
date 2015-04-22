#include "FeatureExtractor.h"
#include "RawData.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "wtime.h"
#include "mathtool.h"
#include "ThreadPool.h"
#include "FeatureExtractorTool.h"

SP_RESULT FeatureExtractor::exFeatures(const RawData *data) {
    return exFeatures(data, \
            sampleRate,
            preEmpFactor, \
            winTime, \
            stepTime, \
            winFunc, \
            minF, \
            maxF, \
            hz2melFunc, \
            mel2hzFunc, \
            nfilts, \
            cepsNum);
}

SP_RESULT FeatureExtractor::exDoubleDeltaFeatures(const RawData *data) {
    exFeatures(data);

    doubleDelta(normalMelCeps);
}
void FeatureExtractor::doubleDelta(std::vector<Feature> & normalMelCeps) {
    int idx, siz = normalMelCeps.size();

    for(idx = 0; idx < siz; idx ++) 
        normalMelCeps[idx].fillDoubleDelta();
}

SP_RESULT FeatureExtractor::exFeatures(const RawData *data, \
        int sampleRate, \
        double preEmpFactor, \
        double winTime, \
        double stepTime, \
        double (*winFunc)(int, int), \
        double minF, \
        double maxF, \
        double (*hz2melFunc)(double), \
        double (*mel2hzFunc)(double), \
        int nfilts, \
        int cepsNum) {
    SP_RESULT res; 
    inital();

    double startT, finishT;
    double totalTime = 0;

    std::cout << "\n  Get the preemphasized data ...\n";

    startT = wtime();
    //res = preEmph(emp_data, data->getData(), data->getFrameNum(), preEmpFactor);
    res = preEmph(e_emp_data, data->getData(), data->getFrameNum(), preEmpFactor);
    finishT = wtime();
    double t_preemp = finishT-startT;
    totalTime += t_preemp;

    std::cout << "  Get the windows ...\n";

    startT = wtime();
    //res = windowing(windows, emp_data, winTime, stepTime, sampleRate, winFunc);
    res = windowing(e_windows, e_emp_data, winTime, stepTime, sampleRate, winFunc);
    finishT = wtime();
    double t_window = finishT-startT;
    totalTime += t_window;

    //startT = wtime();
    //fftPadding(windows);
    //finishT = wtime();
    //double t_fftpad = finishT-startT;
    //totalTime += t_fftpad;

    std::cout << "  Get the Power Spectrum...\n";

    startT = wtime();
    //powSpectrum(powSpec, windows);
    powSpectrum(e_powSpec, e_windows);
    finishT = wtime();
    double t_powSpec = finishT-startT;
    totalTime += t_powSpec;

    //if(powSpec.size() == 0) return SP_SUCCESS;
    
    std::cout << "  Get the MelLog Spectrum...\n";

    int nfft = (e_powFrameSize -1) << 1;

    startT = wtime();
    //fft2MelLog(nfft, melLogSpec, powSpec, nfilts, hz2melFunc, mel2hzFunc, minF, maxF, sampleRate);
    fft2MelLog(nfft, &e_melLogSpec, e_powSpec, nfilts, hz2melFunc, mel2hzFunc, minF, maxF, sampleRate);
    finishT = wtime();
    double t_mel = finishT-startT;
    totalTime += t_mel;

    std::cout << "  Get the DCT MelCepstrum...\n";
    
    startT = wtime();
    //melCepstrum(melCeps, melLogSpec, cepsNum);
    melCepstrum(melCeps, e_melLogSpec, cepsNum);
    finishT = wtime();
    double t_dctCep = finishT-startT;
    totalTime += t_dctCep;

    startT = wtime();
    time_t start = time(0);
    normalization(normalMelCeps, melCeps);
    finishT = wtime();
    double t_norm = finishT-startT;
    totalTime += t_norm;

    //std::cout << "Total Time: " << totalTime << std::endl;
    //std::cout << "PreEmp: " << t_preemp << " s , " << t_preemp*100/totalTime <<"%" <<std::endl;
    //std::cout << "Windowing: " << t_window << " s , " << t_window*100/totalTime <<"%" << std::endl;
    //std::cout << "FFT padding: " << t_fftpad << " s , " << t_fftpad*100/totalTime <<"%"<< std::endl;
    //std::cout << "PowerSpectrum: " << t_powSpec << " s , " << t_powSpec*100/totalTime <<"%"<< std::endl;
    //std::cout << "MelFiltering: " << t_mel << " s , " << t_mel*100/totalTime <<"%"<< std::endl;
    //std::cout << "DCT Ceptrum: " << t_dctCep << " s , " << t_dctCep*100/totalTime <<"%"<< std::endl;
    //std::cout << "Normalization: " << t_norm << " s , " << t_norm*100/totalTime <<"%"<< std::endl;

    return SP_SUCCESS;
}

SP_RESULT FeatureExtractor::normalization(std::vector<Feature> &normalMels, const std::vector<Feature> & melFes) {
    normalMels.clear();
    if(melFes.size() == 0) return SP_SUCCESS;
    
    Feature means, variance;
    int siz = melFes[0].size();
    means.resize(siz);
    variance.resize(siz);
    for(int i = 0;i < siz;i++) {
        means[i] = variance[i] = 0;
    }

    for(int i = 0;i < melFes.size(); i++) {
        for(int j = 0;j < siz; j++) {
            if(melFes[i].size() > j) {
                means[j] += melFes[i][j];

                variance[j] += melFes[i][j] * melFes[i][j];
            }
        }
    }
    for(int i = 0;i < siz;i++) {
        means[i] /= melFes.size();
        variance[i] /= melFes.size();

        variance[i] = sqrt(variance[i]);
    }

    for(int i = 0;i < melFes.size();i++) {
        normalMels.push_back(melFes[i]);
        for(int j = 0;j < siz;j++) {
            if(j < melFes[i].size()) {
                normalMels[i][j] -= means[j];
                normalMels[i][j] /= variance[j];
            }
        }
    }
        
    return SP_SUCCESS;
}

SP_RESULT FeatureExtractor::mel2dct(Feature & feature, std::vector<double> melLog, int cepsNum) {
    int siz = melLog.size();
    feature.resize(siz);
    for(int i = 0;i < siz;i++)
        feature[i] = melLog[i];

//    dct(feature.rawData(), siz, 1);

    dct2(feature.rawData(), siz);

    feature.resize(cepsNum);

    return SP_SUCCESS;
}

SP_RESULT FeatureExtractor::melCepstrum(std::vector<Feature> &cepstrums, \
        const Matrix<double> &melLogSpec, \
        int cepsNum) {
    cepstrums.clear();
    if(melLogSpec.size() == 0) return SP_SUCCESS;

    for(int i = 0;i < melLogSpec[0].size(); i++) {
        std::vector<double> tmp;
        for(int j = 0;j < melLogSpec.size(); j++)
            if(melLogSpec[j].size() > i)
                tmp.push_back(melLogSpec[j][i]);

        cepstrums.push_back(Feature());

        mel2dct(cepstrums[i], tmp, cepsNum);
    }
    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::melCepstrum(std::vector<Feature> &cepstrums, \
        FEATURE_DATA **melLogSpec, \
        int cepsNum) {
    cepstrums.clear();

    //for(int i = 0;i < e_frameNum; i++) {
    //    std::vector<FEATURE_DATA> tmp;
    //    for(int j = 0;j < nfilts; j++)
    //        tmp.push_back(melLogSpec[j][i]);

    //    cepstrums.push_back(Feature());

    //    mel2dct(cepstrums[i], tmp, cepsNum);
    //}
    
    int framePerBlock = 4;
    
    int rowNum = nfilts, 
        colNum = e_frameNum;
    int elementNum = rowNum * colNum; 
    size_t memSize = elementNum*sizeof(FEATURE_DATA);
    
    FEATURE_DATA * r_melLogSpec_data = (FEATURE_DATA *) malloc(memSize);
    FEATURE_DATA ** r_melLogSpec = (FEATURE_DATA **)malloc(colNum * sizeof(FEATURE_DATA *));
    
    for(int i=0; i<colNum; i++){
        r_melLogSpec[i] = &r_melLogSpec_data[i*rowNum];
    }
    reverseMatrix(r_melLogSpec, melLogSpec, rowNum, colNum);
    
    std::cout << "    Finish reverse the melLogSpectrum Matrix. \n";
    for(int i=0; i<colNum; i++){
        //std::cout << "i = "<< i << std::endl;
        for(int j=0; j<rowNum; j++){
            //std::cout << r_melLogSpec[i][j] << " ";
            if(r_melLogSpec[i][j] != melLogSpec[j][i])
                std::cout << "\n    ........Not Equal........!!!!!!\n\n";
        }
        //std::cout << std::endl;
    }
    std::cout << "    colNum: " << colNum << ", rowNum: " << rowNum << std::endl; 
    
    FEATURE_DATA * d_melLogSpec_data;
    
    cudaMalloc((void **) &d_melLogSpec_data, memSize);
    cudaMemcpy(d_melLogSpec_data, r_melLogSpec_data, memSize, cudaMemcpyHostToDevice);

    std::cout << "    Begin to use CUDA now..." << std::endl;

    int blockSize = framePerBlock*rowNum;
    size_t sharedMem = blockSize*sizeof(FEATURE_DATA);
    dim3 dimGrid( ceil((double)elementNum/blockSize) );
    dim3 dimBlock(blockSize);
    mel2dct_cu<<< dimGrid, dimBlock, sharedMem>>>(d_melLogSpec_data, rowNum);
    
    std::cout << "    Finish using CUDA..." << std::endl;
    
    cudaMemcpy(r_melLogSpec_data, d_melLogSpec_data, memSize, cudaMemcpyDeviceToHost);
    
    std::cout << "    Fish cuda copy..." << std::endl;

    for(int i=0; i<colNum; i++){
        Feature tmpFeature;
        tmpFeature.resize(cepsNum);
        for(int j=0; j<cepsNum; j++){
           tmpFeature[j] = r_melLogSpec[i][j]; 
        }
        cepstrums.push_back(tmpFeature);
    }
    
    cudaFree(d_melLogSpec_data);
    free(r_melLogSpec_data);
    free(r_melLogSpec);

    return SP_SUCCESS;
}

SP_RESULT FeatureExtractor::reverseMatrix(FEATURE_DATA **outMatrix, FEATURE_DATA **inMatrix, int rowNum, int colNum){
    for(int i=0; i<colNum; i++){
        for(int j=0; j<rowNum; j++){
            outMatrix[i][j] = inMatrix[j][i];
        }
    }
    return SP_SUCCESS;
}


/*
void FeatureExtractor::fftTask(void *in) {
    fft_task_info * task_info = (fft_task_info *) in;

    std::vector<double> &window = *(task_info->window);
    std::vector<double> &powSpec = *(task_info->powWinSpec);

    windowFFT(powSpec, window);

    delete task_info;
}
*/

//SP_RESULT FeatureExtractor::powSpectrum(Matrix<double> &powSpec, \
//        Matrix<double> &windows) {
//    if(windows.size() == 0) return SP_SUCCESS;
//
//    powSpec.resize(windows.size());
//    
//    int frameNum = windows.size(), 
//        frameSize = windows[0].size(),
//        blockSize = windows[0].size(),
//        elementNum = frameNum * frameSize, 
//        selIdx = (int)(std::log2(frameSize))%2;
//    size_t memSize = elementNum * sizeof(std::complex<double>);
//    size_t sharedMem = 2*blockSize*sizeof(std::complex<double>);
//
//    std::complex<double> *SpeechSignal = new std::complex<double>[elementNum], *d_SpeechSignal;
//    for(int i=0; i<frameNum; i++){
//        int offset = i*frameSize;
//        for(int j=0; j<frameSize; j++){
//            SpeechSignal[offset+j] = std::complex<double>(windows[i][j],0);
//        }
//    }
//
//    double calculationStartT = wtime();
//    double startT, finishT;
//    startT = wtime();
//    //cudaMalloc( (void **) &d_SpeechSignal, memSize*2 );
//    cudaMalloc( (void **) &d_SpeechSignal, memSize );
//
//    cudaMemcpy( d_SpeechSignal, SpeechSignal, memSize, cudaMemcpyHostToDevice);
//    
//    finishT = wtime();
//    std::cout << "Cuda Initialize Time: " << finishT-startT<< std::endl;
//    
//    std::cout << "The select index is: " << selIdx << std::endl;
//
//    dim3 dimGrid( ceil( (double)elementNum/blockSize ) );
//    dim3 dimBlock(blockSize);
//    windowFFT_cu<<< dimGrid, dimBlock, sharedMem >>>(d_SpeechSignal, frameNum, frameSize, 1, selIdx);
//    cudaMemcpy(SpeechSignal, d_SpeechSignal+memSize*selIdx, memSize, cudaMemcpyDeviceToHost);
//    cudaMemcpy(SpeechSignal, d_SpeechSignal, memSize, cudaMemcpyDeviceToHost);
//    
//    double calculationEndT = wtime();
//    printf("PowerSpectrum calculation time: %lf\n", calculationEndT - calculationStartT - (finishT - startT));
//    
//    int resSize=frameSize/2+1, resultOffset;
//    for(int i=0; i<frameNum; i++){
//        powSpec[i].resize(resSize);
//        resultOffset = i*frameSize;
//        for(int j=0; j<resSize; j++)
//            powSpec[i][j] = std::norm(SpeechSignal[resultOffset+j]);
//    }
//
//    cudaFree(d_SpeechSignal);
//    delete []SpeechSignal;
//    /*  
//    for(int i = 0;i < windows.size(); i++) {
//        if(windows[i].size() != siz) continue;
//        windowFFT(powSpec[i], windows[i]);
//    }
//    */
//    
//    /*
//    ThreadPool threadPool(threadNum);
//    for(int i = 0;i < windows.size();i++) {
//        sp_task task;
//
//        if(windows[i].size() != siz) continue;
//
//        fft_task_info *task_info = new fft_task_info;
//        task_info->window = &(windows[i]);
//        task_info->powWinSpec = &(powSpec[i]);
//
//        task.func = fftTask;
//        task.in   = task_info;
//
//        threadPool.addTask(task);
//    }
//    threadPool.run();
//    */
//
//    return SP_SUCCESS;
//}


SP_RESULT FeatureExtractor::powSpectrum(FEATURE_DATA **powSpec, \
        FEATURE_DATA **windows) {
    //if(windows.size() == 0) return SP_SUCCESS;
    
    int frameNum = e_frameNum, 
        frameSize = e_frameSize,
        blockSize = e_frameSize,
        elementNum = frameNum * frameSize, 
        selIdx = (int)(std::log2(frameSize))%2;
    
    //std::cout << "FrameNum: "<< frameNum <<", FrameSize: " << frameSize << ", blockSize: " << blockSize << ", elementNum: " << elementNum << std::endl;
    
    // Memory Size for whole data
    size_t memSize = elementNum * 2 *sizeof(FEATURE_DATA);
    
    // Share Memory Size in the CUDA
    size_t sharedMem = 2 * blockSize * 2 * sizeof(FEATURE_DATA);

    FEATURE_DATA *SpeechSignal_real = new FEATURE_DATA[elementNum*2], 
                 *d_SpeechSignal_real,
                 *d_SpeechSignal_imag;
    FEATURE_DATA *SpeechSignal_imag = &SpeechSignal_real[elementNum];
    
    // Initialize the Speech Signal by windows (imaginary part are all zero)
    memset(SpeechSignal_real, 0, memSize);
    memcpy(SpeechSignal_real, windows[0], memSize/2);
   
    //for(int i=0; i<frameNum; i++){
    //    int beginIdx = i*frameSize;
    //    for(int j=0; j<frameSize; j++){
    //        if(SpeechSignal_real[beginIdx+j] != windows[i][j])
    //            std::cout << "Not equal!!!!!!!!!" << std::endl;
    //    }
    //}

    double startT, finishT, calcStartT, calcEndT;
    calcStartT = startT = wtime();
    
    cudaMalloc( (void **) &d_SpeechSignal_real, memSize );
    cudaMemcpy( d_SpeechSignal_real, SpeechSignal_real, memSize, cudaMemcpyHostToDevice);
    d_SpeechSignal_imag = &d_SpeechSignal_real[elementNum];

    finishT = wtime();
    std::cout << "Cuda Initialize Time: " << finishT-startT<< std::endl;
    std::cout << "The select index is: " << selIdx << std::endl;

    dim3 dimGrid( ceil( (double)elementNum/blockSize ) );
    dim3 dimBlock(blockSize);
    windowFFT_cu<<< dimGrid, dimBlock, sharedMem >>>(d_SpeechSignal_real, d_SpeechSignal_imag, frameNum, frameSize, 1, selIdx);
    cudaMemcpy(SpeechSignal_real, d_SpeechSignal_real, memSize, cudaMemcpyDeviceToHost);
    
    calcEndT = wtime();
    printf("PowerSpectrum calculation time: %lf\n", calcEndT - calcStartT - (finishT - startT));
    
    
    // Calculate the Power Spectrum
    int resSize=frameSize/2+1, frameOffset, finalOffset;
    FEATURE_DATA realPart, imagPart;
    e_powFrameSize = resSize;
    
    e_powSpec = (FEATURE_DATA **) malloc(e_frameNum * sizeof(FEATURE_DATA *));
    FEATURE_DATA *tmp_powSpec = (FEATURE_DATA *) malloc(e_frameNum * resSize * sizeof(FEATURE_DATA));
    
    for(int i=0; i<frameNum; i++){
        e_powSpec[i] = &tmp_powSpec[i*resSize];
        frameOffset = i*frameSize;
        for(int j=0; j<resSize; j++){
            finalOffset = frameOffset + j;
            realPart = SpeechSignal_real[finalOffset];
            imagPart = SpeechSignal_imag[finalOffset];
            e_powSpec[i][j] = realPart*realPart + imagPart*imagPart;
        }
    }

    cudaFree(d_SpeechSignal_real);
    delete []SpeechSignal_real;

    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::getWts(Matrix<double> &wts, \
        int nfft, \
        double minF, \
        double maxF, \
        int sampleRate, \
        int nfilts, \
        double (*hz2melFunc)(double), \
        double (*mel2hzFunc)(double)) {

    int nfreqs = nfft / 2 + 1;
    wts.clear();
    std::vector<double> points;

    double minmel = hz2melFunc(minF);
    double maxmel = hz2melFunc(maxF);
    double step = (maxmel - minmel) / (nfilts + 1);
    for(int i = 0; i <= nfilts + 1; i++) 
        points.push_back(mel2hzFunc( minmel + step * i));

    for(int i = 0; i <= nfilts + 1; i++) {
        points[i] = ceil(points[i] / sampleRate * (nfft - 1));
    }
    for(int i = 0;i < nfilts;i++) {
        wts.push_back(std::vector<double>());

        std::vector<double> &filter = wts[i];

        int lp = points[i], mp = points[i+1], rp = points[i+2];
        double lf = 1.0 * points[i] / nfft * sampleRate;
        double mf = 1.0 * points[i+1] / nfft * sampleRate;
        double rf = 1.0 * points[i+2] / nfft * sampleRate;

        while(filter.size() < lp)
            filter.push_back(0.0);

        for(int k = lp;k <= mp;k++) 
            filter.push_back((1.0*k/nfft * sampleRate - lf) / (mf - lf));

        for(int k = mp+1;k <= rp;k++) 
            filter.push_back((rf - 1.0*k/nfft * sampleRate) / (rf - mf));

        while(filter.size() < nfreqs) 
            filter.push_back(0.0);
    }

    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::getWts(FEATURE_DATA ***p_wts, \
        int nfft, \
        double minF, \
        double maxF, \
        int sampleRate, \
        int nfilts, \
        double (*hz2melFunc)(double), \
        double (*mel2hzFunc)(double)) {

    int nfreqs = nfft / 2 + 1;
    std::vector<double> points;

    FEATURE_DATA ** wts;
    wts = (FEATURE_DATA **) malloc(nfilts*sizeof(FEATURE_DATA *)); 
    size_t memSize = nfilts * nfreqs * sizeof(FEATURE_DATA);
    FEATURE_DATA * wtsData = (FEATURE_DATA *)malloc(memSize); 
    memset(wtsData,0, memSize);

    double minmel = hz2melFunc(minF);
    double maxmel = hz2melFunc(maxF);
    double step = (maxmel - minmel) / (nfilts + 1);
    for(int i = 0; i <= nfilts + 1; i++) 
        points.push_back(mel2hzFunc( minmel + step * i));

    for(int i = 0; i <= nfilts + 1; i++) {
        points[i] = ceil(points[i] / sampleRate * (nfft - 1));
    }
    for(int i = 0;i < nfilts;i++) {
        //wts.push_back(std::vector<double>());
        
        //std::vector<double> &filter = wts[i];
        wts[i] = &wtsData[i*nfreqs];

        int lp = points[i], mp = points[i+1], rp = points[i+2];
        double lf = 1.0 * points[i] / nfft * sampleRate;
        double mf = 1.0 * points[i+1] / nfft * sampleRate;
        double rf = 1.0 * points[i+2] / nfft * sampleRate;

        //while(filter.size() < lp)
        //    filter.push_back(0.0);

        for(int k = lp;k <= mp;k++){ 
            //filter.push_back((1.0*k/nfft * sampleRate - lf) / (mf - lf));
            wts[i][k] = (1.0*k/nfft * sampleRate - lf) / (mf - lf);
        }
        for(int k = mp+1;k <= rp;k++){ 
            //filter.push_back((rf - 1.0*k/nfft * sampleRate) / (rf - mf));
            wts[i][k] = (rf - 1.0*k/nfft * sampleRate) / (rf - mf);
        }
        
        //while(filter.size() < nfreqs) 
        //    filter.push_back(0.0);
    }

    e_filterSize = nfreqs;
    e_melWtsExist = true;
    *p_wts = wts;
    return SP_SUCCESS;
}


/*  
SP_RESULT FeatureExtractor::getMelLog(std::vector<double> & melLog, \
        const std::vector<double> & powSpec, \
        const Matrix<double> &wts) {
    melLog.resize(powSpec.size());
    for(int i = 0;i < melLog.size(); i++) melLog[i] = 0.0;
    for(int i = 0;i < wts.size();i++) {
        int mxSiz = std::min(wts[i].size(), powSpec.size());

        for(int j = 0;j < mxSiz;j++) 
            melLog[j] += powSpec[j] * wts[i][j];
    }
    for(int i = 0;i < melLog.size(); i++) 
        melLog[i] = getDB(melLog[i]);

    return SP_SUCCESS;
}
*/

/*void FeatureExtractor::mulTask(void *in) {
    mul_task_info * task_info = (mul_task_info *) in;

    std::vector<double> &melLog = *(task_info->melLog);
    std::vector<double> &wts = *(task_info->wts);
    Matrix<double> &powSpec = *(task_info->powSpec);

    for(int j = 0;j < powSpec.size();j++) {
        melLog[j] = 0.0;
        int mx = std::min(wts.size(), powSpec[j].size());
        for(int k = 0;k < mx;k++)
            melLog[j] += wts[k] * powSpec[j][k];
    }

    delete task_info;
}*/


//SP_RESULT FeatureExtractor::MatrixMul01(Matrix<double> & melLog, \
//        Matrix<double> &wts, \
//        Matrix<double> & powSpec) {
//    double *h_melLog, *h_wts, *h_powSpec;
//    double *d_melLog, *d_wts, *d_powSpec;
//    
//    size_t memSize1 = wts.size()*powSpec.size()*sizeof(double),
//        memSize2 = wts.size() * wts[0].size()*sizeof(double),
//        memSize3 = powSpec.size() * powSpec[0].size() * sizeof(double);
//    
//    h_melLog = (double *)malloc(memSize1);
//    h_wts = (double *)malloc(memSize2);
//    h_powSpec = (double *)malloc(memSize3);
//    
//    matrix2vector(wts, h_wts);
//    matrix2vector(powSpec, h_powSpec);
//    
//  //  for(int i=0; i<wts.size(); i++){
//  //      int offset = wts[0].size()*i;
//  //      for(int j=0; j<wts[0].size(); j++){
//  //          if(wts[i][j]!=*(h_wts+offset+j))
//  //              std::cout << "\n WTS not equal \n";
//  //      }
//  //  }
//  //  for(int i=0; i<powSpec.size(); i++){
//  //      int offset = powSpec[0].size()*i;
//  //      for(int j=0; j<powSpec[0].size(); j++){
//  //          if(powSpec[i][j]!=*(h_powSpec+offset+j))
//  //              std::cout << "\n powSpec not equal \n";
//  //      }
//  //  }
//  //  
//    double startT = wtime();
//    
//    cudaMalloc((void **)&d_melLog, memSize1);
//    cudaMalloc((void **)&d_wts, memSize2);
//    cudaMalloc((void **)&d_powSpec, memSize3);
//    
//    cudaMemcpy(d_wts, h_wts, memSize2, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_powSpec, h_powSpec, memSize3, cudaMemcpyHostToDevice);
//
//    int bucketNum = (((powSpec.size()-1)/BLOCK_SIZE+1)-1)/COL_STEP+1;
//    int blockNum = (wts.size()-1)/BLOCK_SIZE+1;
//    
//    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//    dim3 dimGrid(bucketNum,blockNum);
//    int r = wts.size(), c = powSpec.size();
//    
//    //printf("%d %d %d\n", r, c, wts[0].size());
//    matrix_mul_kernel<<<dimGrid,dimBlock>>>(d_wts, d_powSpec, d_melLog, r, wts[0].size(), c);
//
//    cudaMemcpy(h_melLog, d_melLog, memSize1, cudaMemcpyDeviceToHost);
//    
//    double endT = wtime();
//    printf("mel filtering calculation time %lf\n", endT-startT);
////    printf("%d %d %d %d\n", wts.size(), wts[0].size(), powSpec[0].size(), powSpec.size());
//    melLog.resize(r);
//    for(int i = 0;i < r;i++)
//        melLog[i].resize(c);
//
//    vector2matrix(h_melLog, melLog);
//    
//    /*
//    for(int i = 0;i < r;i++) {
//        for(int j = 0;j < c;j++) {
//            printf("%lf ", melLog[i][j]); 
//        }
//        puts("");
//    }
//    */
//    
//    
//    /*
//    for(int i = 0;i < r;i++) {
//        for(int j = 0;j < c;j++) {
//            melLog[i][j] = 0.0;
//            int mx = std::min(wts[i].size(), powSpec[j].size());
//            for(int k = 0;k < mx;k++)
//                melLog[i][j] += wts[i][k] * powSpec[j][k];
//        }
//    }
//    */
//
//    free(h_melLog);
//    free(h_wts);
//    free(h_powSpec);
//    cudaFree(d_melLog);
//    cudaFree(d_wts);
//    cudaFree(d_powSpec);
//    
//    return SP_SUCCESS;
//}

SP_RESULT FeatureExtractor::MatrixMul01(FEATURE_DATA ***p_melLog, \
        FEATURE_DATA **wts, \
        FEATURE_DATA **powSpec) {
    FEATURE_DATA *h_melLog, *h_wts, *h_powSpec;
    FEATURE_DATA *d_melLog, *d_wts, *d_powSpec;
    FEATURE_DATA **melLog;
    
    //size_t memSize1 = wts.size()*powSpec.size()*sizeof(double),
    //    memSize2 = wts.size() * wts[0].size()*sizeof(double),
    //    memSize3 = powSpec.size() * powSpec[0].size() * sizeof(double);
    
    size_t memSize1 = nfilts * e_frameNum * sizeof(FEATURE_DATA),
        memSize2 = nfilts * e_filterSize * sizeof(FEATURE_DATA),
        memSize3 = e_frameNum * e_powFrameSize * sizeof(FEATURE_DATA);
    
    //h_melLog = (double *)malloc(memSize1);
    h_melLog = (FEATURE_DATA *)malloc(memSize1);
    
    //h_wts = (double *)malloc(memSize2);
    h_wts = wts[0];
    
    //h_powSpec = (double *)malloc(memSize3);
    h_powSpec = powSpec[0];
    
    
    //matrix2vector(wts, h_wts);
    //matrix2vector(powSpec, h_powSpec);
    
    
  //  for(int i=0; i<wts.size(); i++){
  //      int offset = wts[0].size()*i;
  //      for(int j=0; j<wts[0].size(); j++){
  //          if(wts[i][j]!=*(h_wts+offset+j))
  //              std::cout << "\n WTS not equal \n";
  //      }
  //  }
  //  for(int i=0; i<powSpec.size(); i++){
  //      int offset = powSpec[0].size()*i;
  //      for(int j=0; j<powSpec[0].size(); j++){
  //          if(powSpec[i][j]!=*(h_powSpec+offset+j))
  //              std::cout << "\n powSpec not equal \n";
  //      }
  //  }
  //  
    double startT = wtime();
    
    cudaMalloc((void **)&d_melLog, memSize1);
    cudaMalloc((void **)&d_wts, memSize2);
    cudaMalloc((void **)&d_powSpec, memSize3);
    
    cudaMemcpy(d_wts, h_wts, memSize2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_powSpec, h_powSpec, memSize3, cudaMemcpyHostToDevice);

    //int bucketNum = (((powSpec.size()-1)/BLOCK_SIZE+1)-1)/COL_STEP+1;
    //int blockNum = (wts.size()-1)/BLOCK_SIZE+1;
    
    int bucketNum = (((e_frameNum-1)/BLOCK_SIZE+1)-1)/COL_STEP+1;
    int blockNum = (nfilts-1)/BLOCK_SIZE+1;
    

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(bucketNum,blockNum);
    int r = nfilts, c = e_frameNum;
    
    //printf("%d %d %d\n", r, c, wts[0].size());
    //matrix_mul_kernel<<<dimGrid,dimBlock>>>(d_wts, d_powSpec, d_melLog, r, wts[0].size(), c);
    matrix_mul_kernel<<<dimGrid,dimBlock>>>(d_wts, d_powSpec, d_melLog, r, e_filterSize, c);

    cudaMemcpy(h_melLog, d_melLog, memSize1, cudaMemcpyDeviceToHost);
    
    double endT = wtime();
    printf("mel filtering calculation time %lf\n", endT-startT);
//    printf("%d %d %d %d\n", wts.size(), wts[0].size(), powSpec[0].size(), powSpec.size());
    
    //melLog.resize(r);
    melLog = (FEATURE_DATA **) malloc(nfilts * sizeof(FEATURE_DATA*));
    for(int i = 0;i < r;i++){
        //melLog[i].resize(c);
        melLog[i] = &h_melLog[i*c];
    }
    
    *p_melLog = melLog;
    
    //vector2matrix(h_melLog, melLog);
    
    /*
    for(int i = 0;i < r;i++) {
        for(int j = 0;j < c;j++) {
            printf("%lf ", melLog[i][j]); 
        }
        puts("");
    }
    */
    
    
    /*
    for(int i = 0;i < r;i++) {
        for(int j = 0;j < c;j++) {
            melLog[i][j] = 0.0;
            int mx = std::min(wts[i].size(), powSpec[j].size());
            for(int k = 0;k < mx;k++)
                melLog[i][j] += wts[i][k] * powSpec[j][k];
        }
    }
    */

    cudaFree(d_melLog);
    cudaFree(d_wts);
    cudaFree(d_powSpec);
    
    return SP_SUCCESS;
}


//SP_RESULT FeatureExtractor::fft2MelLog(int nfft, \
//        Matrix<double> &melLog, \
//        Matrix<double> & powSpec, \
//        int nfilts , \
//        double (*hz2melFunc)(double), \
//        double (*mel2hzFunc)(double), \
//        double minF, \
//        double maxF, \
//        int sampleRate) {
//    Matrix<double> wts;
//    
//    double startT, finishT;
//    startT = wtime();
//    getWts(wts, nfft, minF, maxF, sampleRate, nfilts, hz2melFunc, mel2hzFunc);
//    finishT = wtime();
//    std::cout << "getWts: "<<finishT-startT << std::endl;
//    
//    melLog.clear();
//
//    startT = wtime();
//    MatrixMul01(melLog, wts, powSpec);
//    finishT = wtime();
//    std::cout << "MatrixMul: "<<finishT-startT << std::endl;
//
//    startT = wtime();
//    for(int i = 0;i < melLog.size();i++) 
//        for(int j = 0;j < melLog[i].size();j++)
//            melLog[i][j] = log(0.0001+fabs(melLog[i][j]));
//    finishT = wtime();
//    std::cout << "MelLog: "<<finishT-startT << std::endl;
//
//    return SP_SUCCESS;
//}


SP_RESULT FeatureExtractor::fft2MelLog(int nfft, \
        FEATURE_DATA ***p_melLog,
        FEATURE_DATA **powSpec, \
        int nfilts , \
        double (*hz2melFunc)(double), \
        double (*mel2hzFunc)(double), \
        double minF, \
        double maxF, \
        int sampleRate) {
    //Matrix<double> wts;
    
    double startT, finishT;
    startT = wtime();
    std::cout << "    Begin to check if the MelFilter is existed...\n";
    
    if(!e_melWtsExist){
        std::cout << "    Not exist. To get the MelFilter Windows...\n";
        getWts(&e_melWts, nfft, minF, maxF, sampleRate, nfilts, hz2melFunc, mel2hzFunc);
    }
    finishT = wtime();
    std::cout << "getWts: "<<finishT-startT << std::endl;
    //melLog.clear();
    
    //for(int i=0; i<nfilts; i++){
    //    std::cout << "i=" << i << std::endl;
    //    for(int j=0; j<e_filterSize; j++){
    //        std::cout << e_melWts[i][j] << " ";
    //    }
    //    std::cout << std::endl;
    //}


    startT = wtime();
    MatrixMul01(p_melLog, e_melWts, powSpec);
    finishT = wtime();
    std::cout << "MatrixMul: "<<finishT-startT << std::endl;

    FEATURE_DATA **melLog = *p_melLog;
    startT = wtime();
    for(int i = 0;i < nfilts;i++) 
        for(int j = 0;j < e_frameNum;j++){
            //std::cout << " i = "<<i << ", j = "<<j<<"\n";
            melLog[i][j] = log(0.0001+fabs(melLog[i][j]));
        }
    finishT = wtime();
    std::cout << "MelLog: "<<finishT-startT << std::endl;

    return SP_SUCCESS;
}


void FeatureExtractor::windowFFT(std::vector<double> &res, \
        std::vector<double> &data) {
    int blockSize = 256;
    res.resize(data.size() / 2 + 1);
    std::complex<double> *cp = new std::complex<double>[data.size()];
    std::complex<double> *d_cp;

    size_t memSize = data.size()*sizeof(std::complex<double>);
    cudaMalloc((void **) &d_cp, memSize*2+sizeof(int));
    
    for(int i = 0;i < data.size();i++) {
        cp[i] = std::complex<double>(data[i], 0);
    }

    cudaMemcpy(d_cp,cp,memSize, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((double)data.size()/blockSize));
    dim3 dimBlock(blockSize);
    fft_cu_part<<<dimGrid, dimBlock >>>(d_cp, data.size(), 1);
    //fft(cp, data.size(), -1);
    //dft(cp, data.size(), 1);

    int selIdx;
    cudaMemcpy(&selIdx, d_cp+2*data.size(), sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Selected Idx:" <<  selIdx << std::endl;
    cudaMemcpy(cp, d_cp+selIdx*data.size(), memSize, cudaMemcpyDeviceToHost);
    
    for(int i = 0;i < res.size();i++) {
        res[i] = std::norm(cp[i]);
    }

    cudaFree(d_cp);
    delete [] cp;
}

SP_RESULT FeatureExtractor::windowMul(std::vector<double> &window, \
        double (*winFunc)(int, int) ) {
    int M = window.size();
    for(int i = 0;i < M;i++) {
        window[i] *= winFunc(i, M);
    }
    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::windowMul(FEATURE_DATA *window, \
        int size, \
        double (*winFunc)(int, int) ) {
    for(int i = 0;i < size;i++) {
        window[i] *= winFunc(i, size);
    }
    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::windowing(Matrix<double> & out_windows, \
        const std::vector<double> & in, \
        double winTime, \
        double stepTime, \
        int rate, \
        double (*winFunc)(int, int)) {
    int samplePerWin = ceil(winTime * rate);
    int stepPerWin = ceil(stepTime * rate);
    
//    int nfft = 2 ^ (ceil(log(1.0 * samplePerWin)/log(2.0)));
    std::vector<double> buf(samplePerWin);
    for(int i = 0; i < in.size(); i += stepPerWin) {
        for(int j = 0;j < samplePerWin && i+j < in.size(); j++) {
            buf[j] = in[i+j];
        }

        windowMul(buf, winFunc);

        out_windows.push_back(buf);
    }

    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::windowing(FEATURE_DATA **out_windows, \
        const FEATURE_DATA *in, \
        double winTime, \
        double stepTime, \
        int rate, \
        double (*winFunc)(int, int)) {
    int samplePerWin = ceil(winTime * rate);
    int stepPerWin = ceil(stepTime * rate);
    int nfft = (1 << int(ceil(log(1.0 * samplePerWin)/log(2.0))));
    e_frameSize = nfft;
    
    int paddedSize = nfft*ceil((float)size_empData/stepPerWin)*sizeof(FEATURE_DATA);
    FEATURE_DATA *window_data = (FEATURE_DATA *)malloc(paddedSize);
    memset(window_data, 0, paddedSize);
    
    //std::cout << "Padded Size: " << paddedSize << std::endl;
    int cnt=0, i, j, k;
    for(i = 0, k=0; i < size_empData; i += stepPerWin, k += nfft) {
        cnt++;
        for(j = 0;j < samplePerWin && i+j < size_empData; j++) {
            //buf[j] = in[i+j];
            window_data[k+j] = in[i+j];
        }

        //std::cout << "Inner Size: " << j << std::endl;
        //windowMul(buf, winFunc);
        windowMul(&window_data[k],samplePerWin,winFunc);
    }
    
    e_frameNum = cnt;
    e_windows = (FEATURE_DATA **)malloc(cnt*sizeof(FEATURE_DATA *));
    for(i=0,j=0; i<cnt; i++,j+=e_frameSize){
        e_windows[i] = &window_data[j];
    }

    //std::vector<double> buf(samplePerWin);
    //for(int i = 0; i < in.size(); i += stepPerWin) {
    //    for(int j = 0;j < samplePerWin && i+j < in.size(); j++) {
    //        buf[j] = in[i+j];
    //    }

    //    windowMul(buf, winFunc);

    //    out_windows.push_back(buf);
    //}

    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::preEmph(/* out */std::vector<double> &outs, \
        /*in*/const SOUND_DATA* rd, \
        int size, \
        double factor){
    outs.clear();
    outs.push_back(rd[0]);
    for(int i = 1;i<size;i++){
        outs.push_back(1.0 * rd[i] - factor * rd[i-1]);
    }

    return SP_SUCCESS;
}

SP_RESULT FeatureExtractor::preEmph(/* out */FEATURE_DATA *outs, \
        /*in*/const SOUND_DATA* rd, \
        int size, \
        double factor){
    size_empData = size;
    outs[0]=rd[0];
    for(int i = 1;i<size;i++){
        outs[i]=(1.0 * rd[i] - factor * rd[i-1]);
    }

    return SP_SUCCESS;
}
/*
void FeatureExtractor::paddingTask(void *in) {
    padding_task_info * info = (padding_task_info *) in;

    std::vector<double> & window = *(info->window);
    int nfft = info->nfft;

    while(window.size() < nfft) { 
        window.push_back(0.0);
    }

    delete info;
}
*/

SP_RESULT FeatureExtractor::fftPadding(Matrix<double> & windows) {
    if(windows.size() == 0) return SP_SUCCESS;
    int samplePerWin = windows[0].size();

    int nfft = (1 << int(ceil(log(1.0 * samplePerWin)/log(2.0))));

    
    for(int i = 0;i < windows.size();i++) {
        while(windows[i].size() < nfft) 
            windows[i].push_back(0.0);
    }
    
    /*
    ThreadPool threadPool(threadNum);
    for(int i = 0;i < windows.size();i++) {
        struct sp_task task_struct;
        struct padding_task_info *task_info = new padding_task_info;
        
        task_info->window = &(windows[i]);
        task_info->nfft = nfft;

        task_struct.func = paddingTask;
        task_struct.in   = task_info;

        threadPool.addTask(task_struct);
    }
    threadPool.run();
    */
    
    return SP_SUCCESS;
}
