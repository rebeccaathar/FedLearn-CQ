import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import tensorflow as tf

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber, PlotBER
from sionna.utils.metrics import compute_ber



def cdl_channel_user():

    #tf.random.set_seed(client_index)

    # Define the number of UT and BS antennas.
    # For the CDL model, that will be used in this notebook, only
    # a single UT and BS are supported.
    num_ut_ant = 8
    num_bs_ant = 64

    # The number of transmitted streams is equal to the number of UT antennas
    # in both uplink and downlink
    num_streams_per_tx = num_ut_ant

    # Create an RX-TX association matrix
    rx_tx_association = np.array([[1]])

    # Instantiate a StreamManagement object
    # This determines which data streams are determined for which receiver.
    # In this simple setup, this is fairly easy. However, it can get more involved
    # for simulations with many transmitters and receivers.
    sm = StreamManagement(rx_tx_association, num_streams_per_tx)


    rg = ResourceGrid(num_ofdm_symbols=14,
                    fft_size=76,
                    subcarrier_spacing=15e3,
                    num_tx=1,
                    num_streams_per_tx=num_streams_per_tx,
                    cyclic_prefix_length=6,
                    num_guard_carriers=[5,6],
                    dc_null=True,
                    pilot_pattern="kronecker",
                    pilot_ofdm_symbol_indices=[2,11])

    carrier_frequency = 2.6e9 # Carrier frequency in Hz.
                            # This is needed here to define the antenna element spacing.

    ut_array = AntennaArray(num_rows=1,
                            num_cols=int(num_ut_ant/2),
                            polarization="dual",
                            polarization_type="cross",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    bs_array = AntennaArray(num_rows=1,
                            num_cols=int(num_bs_ant/2),
                            polarization="dual",
                            polarization_type="cross",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    direction = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                        # In the `uplink`, the UT is transmitting.

    lower_bound_ns = 45e-9  # 45 ns
    upper_bound_ns = 316e-9  # 316 ns

    # Gerando um número aleatório entre 45 ns e 316 ns
    delay_spread = np.random.uniform(lower_bound_ns, upper_bound_ns)
    print(f'delay spread:  {delay_spread}')
    #delay_spread = 624e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                        # about how to choose this value. 

    cdl_model = "C"       # Suitable values are ["A", "B", "C", "D", "E"]
 
    max_speed = np.random.uniform(low=0, high=50)  # UT speed [m/s].
    min_speed = np.random.uniform(low=0, high=max_speed)  # UT speed [m/s].
    print(max_speed, min_speed)
    # Configure a channel impulse reponse (CIR) generator for the CDL model.
    # cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
    cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=min_speed, max_speed=max_speed)

    # The following values for truncation are recommended.
    # Please feel free to tailor them to you needs.
    l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
    l_tot = l_max-l_min+1

    a, tau = cdl(batch_size=2, num_time_steps=rg.num_time_samples+l_tot-1, sampling_frequency=rg.bandwidth)
    # print("Shape of the path gains: ", a.shape)
    # print("Shape of the delays:", tau.shape)

    channel_time = ApplyTimeChannel(rg.num_time_samples, l_tot=l_tot, add_awgn=True)


    num_bits_per_symbol = 2 # QPSK modulation
    coderate = 0.5 # Code rate
    n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
    k = int(n*coderate) # Number of information bits

    # The binary source will create batches of information bits
    binary_source = BinarySource()

    # The encoder maps information bits to coded bits
    encoder = LDPC5GEncoder(k, n)

    # The mapper maps blocks of information bits to constellation symbols
    mapper = Mapper("qam", num_bits_per_symbol)

    # The resource grid mapper maps symbols onto an OFDM resource grid
    rg_mapper = ResourceGridMapper(rg)

    # The zero forcing precoder precodes the transmit stream towards the intended antennas
    zf_precoder = ZFPrecoder(rg, sm, return_effective_channel=True)

    # OFDM modulator and demodulator
    modulator = OFDMModulator(rg.cyclic_prefix_length)
    demodulator = OFDMDemodulator(rg.fft_size, l_min, rg.cyclic_prefix_length)

    # This function removes nulled subcarriers from any tensor having the shape of a resource grid
    remove_nulled_scs = RemoveNulledSubcarriers(rg)

    # The LS channel estimator will provide channel estimates and error variances
    ls_est = LSChannelEstimator(rg, interpolation_type="nn")

    # The LMMSE equalizer will provide soft symbols together with noise variance estimates
    lmmse_equ = LMMSEEqualizer(rg, sm)

    # The demapper produces LLR for all coded bits
    demapper = Demapper("app", "qam", num_bits_per_symbol)

    # The decoder provides hard-decisions on the information bits
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    batch_size = 4 # We pick a small batch_size as executing this code in Eager mode could consume a lot of memory
    # ebno_db = np.random.randint(-10, 10) #The `Eb/No` value in dB
    ebno_db = np.random.uniform(low=-10, high=10)

    #Computes the Noise Variance (No)
    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
    b = binary_source([batch_size, 1, rg.num_streams_per_tx, encoder.k])
    c = encoder(b)
    x = mapper(c)
    x_rg = rg_mapper(x)

    # The CIR needs to be sampled every 1/bandwith [s].
    # In contrast to frequency-domain modeling, this implies
    # that the channel can change over the duration of a single
    # OFDM symbol. We now also need to simulate more
    # time steps.
    cir = cdl(batch_size, rg.num_time_samples+l_tot-1, rg.bandwidth)

    # OFDM modulation with cyclic prefix insertion
    x_time = modulator(x_rg)

    # Compute the discrete-time channel impulse reponse
    h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min, l_max, normalize=True)

    time_channel = TimeChannel(cdl, rg.bandwidth, rg.num_time_samples,
                                l_min=l_min, l_max=l_max, normalize_channel=True,
                                add_awgn=True, return_channel=True)

    #y_time, h_time = time_channel([x_time, no])
    # Compute the channel output
    y_time = channel_time([x_time, h_time, no])

    # OFDM demodulation and cyclic prefix removal
    y = demodulator(y_time)


    h_hat, err_var = ls_est ([y, no])

    x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
    llr = demapper([x_hat, no_eff])
    b_hat = decoder(llr)
    ber = compute_ber(b, b_hat)

    return ber, ebno_db, no


ber, ebno_db, no = cdl_channel_user()
print("BER: {}".format(ber))
print("Eb/No: {}".format(ebno_db))
print("Noise Level: {}".format(no))