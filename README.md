# Spot-adaptive Knowledge Distillation

## Introduction
This repo benchmarks 11 state-of-the-art knowledge distillation methods with spot-adaptive KD in PyTorch, including: 

- (FitNet) - Fitnets: hints for thin deep nets
- (AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
- (SP) - Similarity-Preserving Knowledge Distillation
- (CC) - Correlation Congruence for Knowledge Distillation
- (VID) - Variational Information Distillation for Knowledge Transfer
- (RKD) - Relational Knowledge Distillation
- (PKT) - Probabilistic Knowledge Transfer for deep representation learning
- (FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer
- (FSP) - A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
- (NST) - Like what you like: knowledge distill via neuron selectivity transfer
- (CRD) - Contrastive Representation Distillation

## Installation
This repo was tested with Ubuntu 16.04.6 LTS, Python 3.6. 
And it should be runnable with PyTorch versions >= 0.4.0.

## Running
1.Fetch the pretrained teacher models by: 
```
sh train_single.sh 
```
which will run the code and save the models to <code> ./run/$dataset/$seed/$model/ckpt </code>

The flags in <code>train_single.sh</code> can be explained as:
- <code>seed</code>: specify the random seed.
- <code>dataset</code>: specify the training dataset.
- <code>num_classes</code>: give the number of categories of the above dataset.
- <code>model</code>: specify the model, see <code>'models/__init__.py'</code> to check the available model types.

Note: the default setting can be seen in config files from <code>'configs/$dataset/seed-$seed/single/$model.yml'</code>. 



2.Run our spot-adaptive KD by:
```
sh train.sh
```


3.(Optional) run the anti spot-adaptive KD by:
```
sh train_anti.sh
```

The flags in <code>train.sh</code> and <code>train_anti.sh</code> can be explained as:
- <code>seed</code>: specify the random seed.
- <code>dataset</code>: specify the training dataset.
- <code>num_classes</code>: give the number of categories of the above dataset.
- <code>net1</code>: specify the teacher model, see <code>'models/__init__.py'</code> to check the available model types.
- <code>net2</code>: specify the student model, see <code>'models/__init__.py'</code> to check the available model types.

Note: the default setting can be seen in config files from <code>'configs/$dataset/seed-$seed/$distiller/$net1-$net2.yml'</code>. 



## Benchmark Results on CIFAR-100
Performance is measured by classification accuracy (%)

- Teacher and student are of the **same** architectural type.


<table>
        <tr>
            <td></td>
            <td colspan="3">resnet56 -> resnet20</td>
            <td colspan="3">resnet110 -> resnet32</td>
            <td colspan="3">resnet32x4 -> resnet8x4</td>
            <td colspan="3">vgg13 -> vgg8</td>
            <td colspan="3">WRN_40_2 -> WRN_16_2</td>
        </tr>
        <tr>
            <td></td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
        </tr>
        <tr>
            <td>T</td>
            <td>72.34</td> <td>-</td> <td>-</td>
            <td>74.31</td> <td>-</td> <td>-</td>
            <td>79.42</td> <td>-</td> <td>-</td>
            <td>74.64</td> <td>-</td> <td>-</td>
            <td>75.61</td> <td>-</td> <td>-</td>
        </tr>
        <tr>
            <td>S</td>
            <td>69.06</td> <td>-</td> <td>-</td>
            <td>71.14</td> <td>-</td> <td>-</td>
            <td>72.50</td> <td>-</td> <td>-</td>
            <td>70.36</td> <td>-</td> <td>-</td>
            <td>73.26</td> <td>-</td> <td>-</td>
        </tr>
        <tr>
            <td>KD</td>
            <td>70.66</td> <td>-</td> <td>-</td>
            <td>73.08</td> <td>-</td> <td>-</td>
            <td>73.33</td> <td>-</td> <td>-</td>
            <td>72.98</td> <td>-</td> <td>-</td>
            <td>74.92</td> <td>-</td> <td>-</td>
        </tr>
        <tr>
            <td>Fitnets</td>
            <td>71.05</td> <td>71.34</td> <td><B>+0.29</B></td>
            <td>73.19</td> <td>73.28</td> <td><B>+0.09</B></td>
            <td>74.46</td> <td>74.60</td> <td><B>+0.14</B></td>
            <td>73.09</td> <td>73.22</td> <td><B>+0.13</B></td>
            <td>75.06</td> <td>75.26</td> <td><B>+0.20</B></td>
        </tr>
        <tr>
            <td>AT</td>
            <td>71.06</td> <td>71.24</td> <td><B>+0.18</B></td>
            <td>73.16</td> <td>73.76</td> <td><B>+0.60</B></td>
            <td>75.16</td> <td>75.42</td> <td><B>+0.26</B></td>
            <td>73.56</td> <td>73.64</td> <td><B>+0.08</B></td>
            <td>75.01</td> <td>75.18</td> <td><B>+0.17</B></td>
        </tr>
        <tr>
            <td>SP</td>
            <td>70.65</td> <td>71.29</td> <td><B>+0.64</B></td>
            <td>72.90</td> <td>72.98</td> <td><B>+0.08</B></td>
            <td>73.93</td> <td>74.33</td> <td><B>+0.40</B></td>
            <td>73.24</td> <td>73.26</td> <td><B>+0.02</B></td>
            <td>73.23</td> <td>74.37</td> <td><B>+1.14</B></td>
        </tr>
        <tr>
            <td>CC</td>
            <td>71.03</td> <td>71.26</td> <td><B>+0.23</B></td>
            <td>73.07</td> <td>73.41</td> <td><B>+0.34</B></td>
            <td>74.60</td> <td>74.63</td> <td><B>+0.03</B></td>
            <td>73.02</td> <td>73.44</td> <td><B>+0.42</B></td>
            <td>74.99</td> <td>75.14</td> <td><B>+0.15</B></td>
        </tr>
        <tr>
            <td>VID</td>
            <td>71.22</td> <td>71.28</td> <td><B>+0.06</B></td>
            <td>73.31</td> <td>73.49</td> <td><B>+0.18</B></td>
            <td>74.57</td> <td>74.88</td> <td><B>+0.31</B></td>
            <td>73.51</td> <td>73.55</td> <td><B>+0.04</B></td>
            <td>74.93</td> <td>75.54</td> <td><B>+0.61</B></td>
        </tr>
        <tr>
            <td>RKD</td>
            <td>71.07</td> <td>71.25</td> <td><B>+0.18</B></td>
            <td>72.91</td> <td>73.50</td> <td><B>+0.59</B></td>
            <td>73.55</td> <td>73.58</td> <td><B>+0.03</B></td>
            <td>73.38</td> <td>73.52</td> <td><B>+0.14</B></td>
            <td>74.46</td> <td>75.50</td> <td><B>+1.04</B></td>
        </tr>
        <tr>
            <td>PKT</td>
            <td>70.72</td> <td>71.42</td> <td><B>+0.70</B></td>
            <td>73.32</td> <td>73.29</td> <td>-0.03</td>
            <td>73.87</td> <td>74.38</td> <td><B>+0.51</B></td>
            <td>73.67</td> <td>73.67</td> <td>-</td>
            <td>75.12</td> <td>75.34</td> <td><B>+0.22</B></td>
        </tr>
        <tr>
            <td>FT</td>
            <td>71.15</td> <td>71.17</td> <td><B>+0.02</B></td>
            <td>73.58</td> <td>73.67</td> <td><B>+0.09</B></td>
            <td>74.60</td> <td>74.75</td> <td><B>+0.15</B></td>
            <td>73.18</td> <td>73.58</td> <td><B>+0.40</B></td>
            <td>75.07</td> <td>75.03</td> <td>-0.04</td>
        </tr>
        <tr>
            <td>NST</td>
            <td>70.68</td> <td>70.77</td> <td><B>+0.09</B></td>
            <td>72.91</td> <td>73.71</td> <td><B>+0.80</B></td>
            <td>74.03</td> <td>74.11</td> <td><B>+0.08</B></td>
            <td>73.33</td> <td>73.40</td> <td><B>+0.07</B></td>
            <td>75.03</td> <td>75.23</td> <td><B>+0.20</B></td>
        </tr>
        <tr>
            <td>CRD</td>
            <td>71.48</td> <td>71.72</td> <td><B>+0.24</B></td>
            <td>73.68</td> <td>73.88</td> <td><B>+0.20</B></td>
            <td>75.28</td> <td>75.75</td> <td><B>+0.47</B></td>
            <td>74.28</td> <td>74.30</td> <td><B>+0.02</B></td>
            <td>75.75</td> <td>75.99</td> <td><B>+0.24</B></td>
        </tr>
</table>



- Teacher and student are of **different** architectural type.

<table>
        <tr>
            <td></td>
            <td colspan="3">vgg13 -> MobileNetV2</td>
            <td colspan="3">ResNet50 -> MobileNetV2</td>
            <td colspan="3">ResNet50 -> vgg8</td>
            <td colspan="3">resnet32x4 -> resnet32</td>
            <td colspan="3">WRN_40_2 -> ShuffleNetV1</td>
        </tr>
        <tr>
            <td></td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
        </tr>
        <tr>
            <td>T</td>
            <td>74.64</td> <td>-</td> <td>-</td>
            <td>79.34</td> <td>-</td> <td>-</td>
            <td>79.34</td> <td>-</td> <td>-</td>
            <td>74.92</td> <td>-</td> <td>-</td>
            <td>75.61</td> <td>-</td> <td>-</td>
        </tr>
        <tr>
            <td>S</td>
            <td>64.60</td> <td>-</td> <td>-</td>
            <td>64.60</td> <td>-</td> <td>-</td>
            <td>70.36</td> <td>-</td> <td>-</td>
            <td>71.14</td> <td>-</td> <td>-</td>
            <td>70.50</td> <td>-</td> <td>-</td>
        </tr>
        <tr>
            <td>KD</td>
            <td>67.37</td> <td>-</td> <td>-</td>
            <td>67.35</td> <td>-</td> <td>-</td>
            <td>73.81</td> <td>-</td> <td>-</td>
            <td>72.98</td> <td>-</td> <td>-</td>
            <td>74.83</td> <td>-</td> <td>-</td>
        </tr>
        <tr>
            <td>Fitnets</td>
            <td>72.63</td> <td>72.93</td> <td><B>+0.30</B></td>
            <td>72.96</td> <td>73.35</td> <td><B>+0.39</B></td>
            <td>73.24</td> <td>73.78</td> <td><B>+0.54</B></td>
            <td>72.07</td> <td>72.56</td> <td><B>+0.49</B></td>
            <td>75.21</td> <td>75.46</td> <td><B>+0.25</B></td>
        </tr>
        <tr>
            <td>AT</td>
            <td>72.12</td> <td>72.51</td> <td><B>+0.39</B></td>
            <td>71.82</td> <td>72.30</td> <td><B>+0.48</B></td>
            <td>73.60</td> <td>73.61</td> <td><B>+0.01</B></td>
            <td>72.85</td> <td>72.92</td> <td><B>+0.07</B></td>
            <td>76.35</td> <td>76.55</td> <td><B>+0.20</B></td>
        </tr>
        <tr>
            <td>SP</td>
            <td>72.96</td> <td>72.76</td> <td>-0.20</td>
            <td>73.17</td> <td>73.29</td> <td><B>+0.12</B></td>
            <td>73.71</td> <td>74.21</td> <td><B>+0.50</B></td>
            <td>72.12</td> <td>72.38</td> <td><B>+0.26</B></td>
            <td>76.27</td> <td>76.76</td> <td><B>+0.49</B></td>
        </tr>
        <tr>
            <td>CC</td>
            <td>72.73</td> <td>72.78</td> <td><B>+0.05</B></td>
            <td>72.61</td> <td>72.70</td> <td><B>+0.09</B></td>
            <td>73.20</td> <td>73.88</td> <td><B>+0.68</B></td>
            <td>72.44</td> <td>72.74</td> <td><B>+0.30</B></td>
            <td>75.24</td> <td>75.78</td> <td><B>+0.54</B></td>
        </tr>
        <tr>
            <td>VID</td>
            <td>72.10</td> <td>71.86</td> <td><B>+0.76</B></td>
            <td>73.00</td> <td>73.10</td> <td><B>+0.10</B></td>
            <td>73.24</td> <td>73.56</td> <td><B>+0.32</B></td>
            <td>72.29</td> <td>72.67</td> <td><B>+0.38</B></td>
            <td>76.03</td> <td>76.22</td> <td><B>+0.19</B></td>
        </tr>
        <tr>
            <td>RKD</td>
            <td>72.58</td> <td>72.85</td> <td><B>+0.27</B></td>
            <td>73.05</td> <td>73.39</td> <td><B>+0.34</B></td>
            <td>73.36</td> <td>73.57</td> <td><B>+0.21</B></td>
            <td>71.54</td> <td>72.11</td> <td><B>+0.57</B></td>
            <td>75.96</td> <td>75.47</td> <td>-0.49</td>
        </tr>
        <tr>
            <td>PKT</td>
            <td>72.76</td> <td>72.79</td> <td><B>+0.03</B></td>
            <td>72.99</td> <td>73.31</td> <td><B>+0.32</B></td>
            <td>73.39</td> <td>73.96</td> <td><B>+0.57</B></td>
            <td>72.04</td> <td>72.45</td> <td><B>+0.41</B></td>
            <td>75.66</td> <td>75.94</td> <td><B>+0.28</B></td>
        </tr>
        <tr>
            <td>FT</td>
            <td>71.99</td> <td>72.36</td> <td><B>+0.37</B></td>
            <td>72.85</td> <td>72.91</td> <td><B>+0.06</B></td>
            <td>73.11</td> <td>73.68</td> <td><B>+0.57</B></td>
            <td>72.42</td> <td>72.73</td> <td><B>+0.31</B></td>
            <td>76.05</td> <td>75.73</td> <td>-0.32</td>
        </tr>
        <tr>
            <td>NST</td>
            <td>72.19</td> <td>72.20</td> <td><B>+0.01</B></td>
            <td>73.01</td> <td>73.03</td> <td><B>+0.02</B></td>
            <td>-</td> <td>-</td> <td>-</td>
            <td>72.34</td> <td>72.74</td> <td><B>+0.40</B></td>
            <td>75.88</td> <td>75.73</td> <td>-0.15</td>
        </tr>
        <tr>
            <td>CRD</td>
            <td>73.56</td> <td>73.82</td> <td><B>+0.26</B></td>
            <td>73.99</td> <td>74.03</td> <td><B>+0.04</B></td>
            <td>74.14</td> <td>74.46</td> <td><B>+0.32</B></td>
            <td>73.29</td> <td>73.40</td> <td><B>+0.11</B></td>
            <td>76.17</td> <td>76.72</td> <td><B>+0.55</B></td>
        </tr>
</table>

## Benchmark Results on Tiny-ImageNet
Performance is measured by classification accuracy (%)

<table>
        <tr>
            <td></td>
            <td colspan="3">resnet56 -> resnet20</td>
            <td colspan="3">resnet110 -> resnet20</td>
            <td colspan="3">vgg13 -> vgg8</td>
            <td colspan="3">WRN_40_2 -> WRN_16_2</td>
            <td colspan="3">vgg13 -> MobileNetV2</td>
        </tr>
        <tr>
            <td></td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
            <td>Stan.</td> <td>Adap.</td> <td>&Delta;</td>
        </tr>
        <tr>
            <td>T</td>
            <td>58.34</td> <td>-</td> <td>-</td>
            <td>58.46</td> <td>-</td> <td>-</td>
            <td>60.09</td> <td>-</td> <td>-</td>
            <td>61.26</td> <td>-</td> <td>-</td>
            <td>60.09</td> <td>-</td> <td>-</td>
        </tr>
        <tr>
            <td>S</td>
            <td>52.66</td> <td>-</td> <td>-</td>
            <td>51.89</td> <td>-</td> <td>-</td>
            <td>56.03</td> <td>-</td> <td>-</td>
            <td>57.17</td> <td>-</td> <td>-</td>
            <td>57.73</td> <td>-</td> <td>-</td>
        </tr>
        <tr>
            <td>KD</td>
            <td>53.04</td> <td>-</td> <td>-</td>
            <td>53.40</td> <td>-</td> <td>-</td>
            <td>57.33</td> <td>-</td> <td>-</td>
            <td>59.16</td> <td>-</td> <td>-</td>
            <td>60.02</td> <td>-</td> <td>-</td>
        </tr>
        <tr>
            <td>Fitnets</td>
            <td>54.50</td> <td>54.59</td> <td><B>+0.09</B></td>
            <td>54.04</td> <td>54.77</td> <td><B>+0.73</B></td>
            <td>58.05</td> <td>59.05</td> <td><B>+1.00</B></td>
            <td>58.88</td> <td>59.02</td> <td><B>+0.14</B></td>
            <td>61.37</td> <td>61.57</td> <td><B>+0.20</B></td>
        </tr>
        <tr>
            <td>AT</td>
            <td>54.33</td> <td>55.39</td> <td><B>+1.06</B></td>
            <td>54.00</td> <td>54.47</td> <td><B>+0.47</B></td>
            <td>58.75</td> <td>58.69</td> <td>-0.06</td>
            <td>58.56</td> <td>58.72</td> <td><B>+0.16</B></td>
            <td>60.84</td> <td>61.22</td> <td><B>+0.38</B></td>
        </tr>
        <tr>
            <td>SP</td>
            <td>53.86</td> <td>54.69</td> <td><B>+0.83</B></td>
            <td>53.92</td> <td>54.39</td> <td><B>+0.47</B></td>
            <td>58.78</td> <td>58.83</td> <td><B>+0.05</B></td>
            <td>56.78</td> <td>57.65</td> <td><B>+0.87</B></td>
            <td>59.64</td> <td>61.50</td> <td><B>+1.86</B></td>
        </tr>
        <tr>
            <td>CC</td>
            <td>54.03</td> <td>55.02</td> <td><B>+0.99</B></td>
            <td>54.26</td> <td>54.69</td> <td><B>+0.43</B></td>
            <td>58.18</td> <td>58.67</td> <td><B>+0.49</B></td>
            <td>59.04</td> <td>59.08</td> <td><B>+0.04</B></td>
            <td>61.87</td> <td>61.68</td> <td>-0.19</td>
        </tr>
        <tr>
            <td>VID</td>
            <td>53.25</td> <td>53.54</td> <td><B>+0.29</B></td>
            <td>53.94</td> <td>54.34</td> <td><B>+0.40</B></td>
            <td>58.55</td> <td>59.16</td> <td><B>+0.61</B></td>
            <td>58.78</td> <td>59.35</td> <td><B>+0.57</B></td>
            <td>61.34</td> <td>61.47</td> <td><B>+0.13</B></td>
        </tr>
        <tr>
            <td>RKD</td>
            <td>53.68</td> <td>54.18</td> <td><B>+0.50</B></td>
            <td>53.88</td> <td>54.23</td> <td><B>+0.35</B></td>
            <td>58.58</td> <td>58.64</td> <td><B>+0.06</B></td>
            <td>59.31</td> <td>59.43</td> <td><B>+0.12</B></td>
            <td>61.45</td> <td>61.84</td> <td><B>+0.39</B></td>
        </tr>
        <tr>
            <td>PKT</td>
            <td>54.20</td> <td>54.70</td> <td><B>+0.50</B></td>
            <td>54.00</td> <td>54.40</td> <td><B>+0.40</B></td>
            <td>58.96</td> <td>59.00</td> <td><B>+0.04</B></td>
            <td>58.39</td> <td>58.83</td> <td><B>+0.44</B></td>
            <td>62.07</td> <td>62.23</td> <td><B>+0.16</B></td>
        </tr>
        <tr>
            <td>FT</td>
            <td>54.58</td> <td>54.62</td> <td><B>+0.04</B></td>
            <td>53.59</td> <td>53.91</td> <td><B>+0.32</B></td>
            <td>58.87</td> <td>58.83</td> <td>-0.04</td>
            <td>58.72</td> <td>59.36</td> <td><B>+0.64</B></td>
            <td>61.49</td> <td>61.60</td> <td><B>+0.11</B></td>
        </tr>
        <tr>
            <td>NST</td>
            <td>53.66</td> <td>54.21</td> <td><B>+0.55</B></td>
            <td>53.82</td> <td>54.33</td> <td><B>+0.51</B></td>
            <td>58.32</td> <td>58.55</td> <td><B>+0.23</B></td>
            <td>57.76</td> <td>58.80</td> <td><B>+1.04</B></td>
            <td>60.56</td> <td>60.89</td> <td><B>+0.33</B></td>
        </tr>
        <tr>
            <td>CRD</td>
            <td>55.04</td> <td>55.21</td> <td><B>+0.17</B></td>
            <td>54.83</td> <td>55.63</td> <td><B>+0.80</B></td>
            <td>58.88</td> <td>59.27</td> <td><B>+0.39</B></td>
            <td>59.42</td> <td>59.81</td> <td><B>+0.39</B></td>
            <td>61.63</td> <td>62.73</td> <td><B>+1.10</B></td>
        </tr>
</table>
