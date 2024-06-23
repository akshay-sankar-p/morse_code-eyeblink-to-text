import streamlit as st



st.markdown("<div style='background-color: #333333;border: 3px solid #219C90; border-radius:100px;'><h1 style='text-align:center;color:white;'>TALK WITH HANDS</h1></div",unsafe_allow_html=True)


def page_home():
    st.title("Introduction")
    st.write("Welcome to the my web page!")

    st.subheader('What is sign languge?')

    st.write('-Juan Pablo de Bonet is credited with publishing the first sign language instructional book for the deaf in 1620. The book was based on the work of Girolamo Cardano, an Italian physician, who believed that it wasn’t necessary to hear words in order to understand ideas.To clarify, there is a big difference between ASL as a language versus signed English. Those who speak ASL fluently use their eyes, hands, face and body. The vocabulary and grammar of ASL is also different from English. As a result, learning to speak ASL as a language will be more demanding than just learning to communicate with signs and fingerspelling.')
    st.write()
    st.write()
    st.write()
    st.subheader('Main Advantages')

    if st.button('Advantages'):
        st.image('news.jpg',use_column_width=True)
        st.image('doctor.jpg',use_column_width=True)
        st.image('army.jpg',use_column_width=True)
        st.image('deaf.jpg',use_column_width=True)


def page_about():
    import logging
    import logging.handlers
    import queue
    import time
    import urllib.request
    import os
    from pathlib import Path

    import numpy as np
    import pydub
    import streamlit as st
    from twilio.rest import Client
    import cv2
    import mediapipe as mp
    import numpy as np
    import streamlit as st
    import av
    import io


    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    selected_value = st.slider("Select number of words to translate", min_value=0, max_value=15, value=1, step=1)

    HERE = Path(__file__).parent

    logger = logging.getLogger(__name__)
    have = os.listdir('SL')
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
    def download_file(url, download_to: Path, expected_size=None):
        # Don't download the file twice.
        # (If possible, verify the download using the file length.)
        if download_to.exists():
            if expected_size:
                if download_to.stat().st_size == expected_size:
                    return
            else:
                st.info(f"{url} is already downloaded.")
                if not st.button("Download again?"):
                    return

        download_to.parent.mkdir(parents=True, exist_ok=True)

        # These are handles to two visual elements to animate.
        weights_warning, progress_bar = None, None
        try:
            weights_warning = st.warning("Downloading %s..." % url)
            progress_bar = st.progress(0)
            with open(download_to, "wb") as output_file:
                with urllib.request.urlopen(url) as response:
                    length = int(response.info()["Content-Length"])
                    counter = 0.0
                    MEGABYTES = 2.0 ** 20.0
                    while True:
                        data = response.read(8192)
                        if not data:
                            break
                        counter += len(data)
                        output_file.write(data)

                        # We perform animation by overwriting the elements.
                        weights_warning.warning(
                            "Downloading %s... (%6.2f/%6.2f MB)"
                            % (url, counter / MEGABYTES, length / MEGABYTES)
                        )
                        progress_bar.progress(min(counter / length, 1.0))
        # Finally, we remove these visual elements by calling .empty().
        finally:
            if weights_warning is not None:
                weights_warning.empty()
            if progress_bar is not None:
                progress_bar.empty()


    # This code is based on https://github.com/whitphx/streamlit-webrtc/blob/c1fe3c783c9e8042ce0c95d789e833233fd82e74/sample_utils/turn.py
    @st.cache_data  # type: ignore
    def get_ice_servers():
        """Use Twilio's TURN server because Streamlit Community Cloud has changed
        its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
        We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
        but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
        See https://github.com/whitphx/streamlit-webrtc/issues/1213
        """

        # Ref: https://www.twilio.com/docs/stun-turn/api
        try:
            account_sid = os.environ["TWILIO_ACCOUNT_SID"]
            auth_token = os.environ["TWILIO_AUTH_TOKEN"]
        except KeyError:
            logger.warning(
                "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
            )
            return [{"urls": ["stun:stun.l.google.com:19302"]}]

        client = Client(account_sid, auth_token)

        token = client.tokens.create()

        return token.ice_servers

    





    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)


    st.header(" Speech-to-Sign")
        

    # https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
    MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
    LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
    MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
    LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
    download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)

    lm_alpha = 0.931289039105002
    lm_beta = 1.1834137581510284
    beam = 100

    sound_only_page = "convert speech to sign"
    app_mode = st.selectbox("Choose the app mode", [sound_only_page])
    stream=None
    if app_mode == sound_only_page:
        
    
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": False, "audio": True},
        )

        status_indicator = st.empty()

        if not webrtc_ctx.state.playing:
            status_indicator.write("Loading...")
            text_output = st.empty()
            stream = None
    cnt=0   
        
    o=[]
    
    if webrtc_ctx.audio_receiver:
        with st.spinner('In progress...'):
            while True:
                if stream is None:
                    from deepspeech import Model

                    model = Model(str(MODEL_LOCAL_PATH))
                    model.enableExternalScorer(str(LANG_MODEL_LOCAL_PATH))
                    model.setScorerAlphaBeta(lm_alpha, lm_beta)
                    model.setBeamWidth(beam)

                    stream = model.createStream()

                    status_indicator.write("Model loaded.")

                sound_chunk = pydub.AudioSegment.empty()
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                except queue.Empty:
                    time.sleep(0.1)
                    status_indicator.write("No frame arrived.")
                    continue

                status_indicator.write("Running. Say something!")
                for audio_frame in audio_frames:
                    sound = pydub.AudioSegment(
                        data=audio_frame.to_ndarray().tobytes(),
                        sample_width=audio_frame.format.bytes,
                        frame_rate=audio_frame.sample_rate,
                        channels=len(audio_frame.layout.channels),
                    )
                    sound_chunk += sound

                if len(sound_chunk) > 0:
                    sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                        model.sampleRate()
                    )
                    buffer = np.array(sound_chunk.get_array_of_samples())
                    stream.feedAudioContent(buffer)
                    text = stream.intermediateDecode()
                    up=text.split()
                    filtered_words = [word for word in up if word.lower() in have]
                
                    o.extend(filtered_words)
                    if len(o)>=selected_value:
                        st.write('Words Recieved',len(o))
                        
                        yy=[]
                        for word in o:
                            dire = os.path.join('SL', word)
            
                            files = os.listdir(dire)
                
                            if files:
                                sign = files[0]
                                ved=os.path.join(dire, sign)
                                cap = cv2.VideoCapture(ved)

                                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                                    while cap.isOpened():
                                        ret, image = cap.read()
                                        if not ret:
                                            break
                                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                        height, width, _ = image.shape
                                        blank = np.zeros_like(image)

                                        results = pose.process(image)
                                        mp_drawing.draw_landmarks(blank, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                                        mp_drawing.draw_landmarks(blank, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=None)
                                        hand_results = hands.process(image)
                                        if hand_results.multi_hand_landmarks:
                                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                                mp_drawing.draw_landmarks(blank, hand_landmarks, mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=None)
                                            yy.append(blank)
                        st.write('just wait a little longer!😄')
                        n_frmaes = len(yy) 

                        width, height, fps = 500, 300, len(yy)*30  

                        output_memory_file = io.BytesIO() 

                        output = av.open(output_memory_file, 'w', format="mp4")  
                        stream = output.add_stream('h264', str(fps))  
                        stream.width = width  
                        stream.height = height
                        stream.pix_fmt = 'yuv420p'  
                        stream.options = {'crf': '17'}  

                        for h in yy:
                            for i in range(n_frmaes):
                                img =h   
                                frame = av.VideoFrame.from_ndarray(img)  
                                packet = stream.encode(frame) 
                                output.mux(packet) 
                        packet = stream.encode(None)
                        output.mux(packet)
                        output.close()

                        output_memory_file.seek(0) 
                        st.video(output_memory_file) 
                        set_string = ', '.join(map(str, o))
                        st.write(f'you are saying  - {set_string} - by hands')
                        st.success('Process completed!')
                        break

def mat():
    import cv2
    import mediapipe as mp
    import numpy as np
    import streamlit as st
    import av
    import io
    import os

    yy=[]
    st.header('Text-sign')

    texx = st.text_input('Enter your text here')

    have = os.listdir('SL')

    input_words = texx.split()
    filtered_words = [word for word in input_words if word.lower() in have]

    filtered_sentence = ' '.join(filtered_words)

    if len(texx) != len(filtered_sentence):
        st.write('Due to some practical issue, some words have been dropped:')
        st.write(filtered_sentence)
    else:
        st.write(f'your input text:{texx}')

    if filtered_sentence and st.button('Convert words to signs'):
        for word in filtered_words:
            dire = os.path.join('SL', word)
        
            files = os.listdir(dire)
            
            if files:
                # Get the first file (assuming it's the sign)
                sign = files[0]
                ved=os.path.join(dire, sign)
            else:
                st.write(f"No sign found for '{word}'")

            

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_holistic = mp.solutions.holistic
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

            

            cap = cv2.VideoCapture(ved)

            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                with st.spinner('In pregress...'):
                    while cap.isOpened():
                        ret, image = cap.read()
                        if not ret:
                            break
                        
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        height, width, _ = image.shape
                        blank = np.zeros_like(image)

                        results = pose.process(image)
                        mp_drawing.draw_landmarks(blank, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(blank, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=None)
                        
                        hand_results = hands.process(image)
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                    
                                    mp_drawing.draw_landmarks(blank, hand_landmarks, mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=None)
                                    
                            yy.append(blank)
        n_frmaes = len(yy)  # Select number of frames (for testing).

        width, height, fps = 500, 300, len(yy)*30  # Select video resolution and framerate.

        output_memory_file = io.BytesIO()  # Create BytesIO "in memory file".

        output = av.open(output_memory_file, 'w', format="mp4")  # Open "in memory file" as MP4 video output
        stream = output.add_stream('h264', str(fps))  # Add H.264 video stream to the MP4 container, with framerate = fps.
        stream.width = width  # Set frame width
        stream.height = height  # Set frame height
        #stream.pix_fmt = 'yuv444p'   # Select yuv444p pixel format (better quality than default yuv420p).
        stream.pix_fmt = 'yuv420p'   # Select yuv420p pixel format for wider compatibility.
        stream.options = {'crf': '17'}  # Select low crf for high quality (the price is larger file size).




        for h in yy:
            for i in range(n_frmaes):
                img =h   # Create OpenCV image for testing (resolution 192x108, pixel format BGR).
                frame = av.VideoFrame.from_ndarray(img)  # Convert image from NumPy Array to frame.
                packet = stream.encode(frame)  # Encode video frame
                output.mux(packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).

        # Flush the encoder
        packet = stream.encode(None)
        output.mux(packet)
        output.close()

        output_memory_file.seek(0)
        
        st.video(output_memory_file) 
    
def bb():
    
    import streamlit as st
    import cv2
    import mediapipe as mp

    pp=[[143.2422128119046,
    124.97092304151649,
    157.9387993268234,
    151.27281282906037,
    174.06876761490028,
    174.73678268509903,
    169.35738975160336,
    178.53546342344873,
    171.13508938336662,
    143.8861712295243,
    44.67363690242539,
    156.49086845154136,
    150.77562988154762,
    30.11859864710275,
    165.66367755531178,
    38.41197917178187,
    80.69161034655988,
    80.86614092750386,
    121.07262764738834,
    87.80277962151406,
    99.06846817780423,
    79.5193785405615],
    [160.31910482683264,
    168.2891887301983,
    169.5007953360304,
    152.36272521569148,
    125.53679840954526,
    155.04469614280225,
    155.17606312804605,
    109.14842376588655,
    137.1993709957673,
    160.80163862449402,
    147.74749959188142,
    166.51300580145204,
    157.28955155649942,
    169.7613901579168,
    177.37517381379652,
    15.007807726870505,
    104.9700424336669,
    76.04641402629275,
    109.18581796623472,
    68.12165313040065,
    112.63682839999451,
    74.76841515323088],
    [171.6902722306973,
    161.70683883238357,
    175.2544964591216,
    118.51500159992592,
    170.74860358243808,
    172.66845070546506,
    112.28736430414085,
    170.1243500888145,
    154.76978713568852,
    94.79048599906449,
    160.5179962287055,
    117.63684643957801,
    95.32837718236392,
    163.66398920952838,
    120.43806955182296,
    14.206433232202825,
    119.67336509967791,
    65.41871021194123,
    110.48494185588777,
    88.20614352833991,
    90.44462126651518,
    90.23118260336413],
    [164.10171971077528,
    162.4132342403322,
    169.79118583928383,
    156.75569334365656,
    31.438160723597612,
    174.16074377966262,
    158.79669697764956,
    26.808405077811877,
    171.34997651675448,
    156.86276952226524,
    23.3332361396318,
    164.839498396938,
    154.02566140396917,
    29.78984886931884,
    162.68209721842635,
    16.16606516872908,
    109.42810236130981,
    74.45853935355076,
    113.3211798720174,
    69.5781749044682,
    117.2593598387585,
    53.5133503057742],
    [157.37361472243919,
    164.97497638799723,
    175.77925535385367,
    153.60630814229927,
    30.348901769467727,
    169.67674401480994,
    153.290851695612,
    28.67180245094567,
    161.49823922305262,
    152.33482717969744,
    20.438490596118616,
    155.155355232513,
    153.3236500663902,
    19.673427163583334,
    159.10930111863044,
    26.42938113850044,
    107.37988226834317,
    71.65700476933839,
    115.84043521338833,
    65.70958703899059,
    119.95917916560737,
    51.95604333692273],
    [143.7846377851781,
    140.5835692415193,
    170.51538401029762,
    156.98417136851012,
    145.2976116328046,
    141.27695273031853,
    131.86261661954813,
    51.55527344223359,
    154.00877289837567,
    125.34997574994891,
    52.03892877862289,
    145.30328306639572,
    144.41825996396489,
    32.76906076330937,
    149.22002498311716,
    44.697064069133155,
    104.32225151794225,
    99.893431673346,
    80.73112544237017,
    101.19504334450788,
    83.75264519695108,
    96.55014954388513],
    [151.8407435818444,
    171.7086824829632,
    177.00309767356407,
    169.00083296384383,
    139.66810729362868,
    160.5506402246076,
    169.20494684933504,
    138.5274524034801,
    150.93864749142188,
    167.98457617264188,
    138.551317777956,
    159.3619204462399,
    166.80000854294119,
    147.11552471196657,
    158.03341973846923,
    55.20955326754469,
    97.06733402129323,
    84.13351753980726,
    103.0869056073615,
    77.65708545862437,
    100.95208439192729,
    80.22465435124155],
    [166.2346725733711,
    161.56208716469175,
    142.18806842300043,
    166.84906482613155,
    142.24680939528596,
    174.88425984773465,
    173.63822681532065,
    166.72861111451212,
    175.47239847529806,
    173.9007065299861,
    170.49575440357367,
    175.9280455515021,
    176.10066573660703,
    177.02407361984714,
    175.1399505928178,
    27.855974474501675,
    99.81646322509464,
    80.1504680864518,
    120.5277735110563,
    62.620134282385145,
    133.05930944972565,
    53.92732771307506],
    [166.59844354835047,
    177.20532804650475,
    172.84485049005303,
    140.1772423806255,
    49.62073663127825,
    170.99371833867173,
    141.03019777076054,
    52.89775992626484,
    172.68572992399749,
    149.02594957326892,
    44.180606106098196,
    170.9318689798394,
    167.9507555089273,
    160.26052333853002,
    170.79572139666524,
    23.06796606869042,
    97.2518538837486,
    83.98565289579997,
    97.78483380625914,
    82.08517292030415,
    104.30122226218333,
    85.77342777725639],
    [161.3146802533321,
    134.3786519757711,
    165.48305867218414,
    163.8794915831539,
    175.86952860253035,
    172.54637390240603,
    163.5281934257304,
    177.14905072017274,
    170.95857135158775,
    137.79336405825958,
    47.90907295111305,
    156.48676681817167,
    138.58627999184657,
    38.75391847553451,
    158.4863802976734,
    21.01977067594214,
    125.24460340111246,
    76.75858517833406,
    125.7446427925495,
    68.04901581574453,
    117.51582894458038,
    57.031698877429974],
    [156.7709909027392,
    152.65675281218125,
    167.19951558925646,
    154.58659637121937,
    131.84099140033,
    154.5083140288787,
    136.27392132216298,
    118.31885998616555,
    150.8215134779364,
    118.74265130489628,
    124.62152183001207,
    163.28317812624252,
    86.09790317351714,
    161.410134784948,
    165.2493153657144,
    45.36088057490822,
    119.06178685926011,
    63.336876550404334,
    130.4370406713108,
    56.14744387175166,
    125.03592590308163,
    77.8327890109254],
    [165.48111579482747,
    169.30491398515895,
    178.695424492331,
    165.72253382297393,
    177.47540483813577,
    176.4348018225828,
    127.45459454635416,
    160.77689524531192,
    161.12431255265622,
    90.60928575479163,
    84.13985157312307,
    155.1933641879901,
    85.11116527569996,
    88.21590647672977,
    142.30678687812085,
    19.286162154293123,
    109.66951726644432,
    55.52007250053999,
    128.43851849437243,
    66.61952196767213,
    102.40489286217934,
    74.82079993203455],
    [163.47701953367832,
    172.63610582588748,
    147.7963650986148,
    136.97188002053565,
    92.90719146520709,
    157.59852632354315,
    132.39712022076438,
    87.53842381067092,
    158.4825016090973,
    129.2890093657461,
    92.07108400731748,
    157.13422743730388,
    128.70970929274358,
    105.73344716499898,
    154.1712379381518,
    41.90934728671603,
    88.4180776981091,
    85.86815901522625,
    99.33420517451094,
    80.94817815924019,
    99.23484674808594,
    79.407556163647],
    [145.37526107618018,
    140.7674180805039,
    171.36180006100807,
    148.71470967654886,
    57.99570411395125,
    149.62855338335552,
    148.0500949969217,
    37.07743418520152,
    157.45719005689662,
    127.90902890167804,
    46.261438771609384,
    144.89139220365465,
    127.37169500654912,
    38.30062037201681,
    140.7287870800916,
    34.98239287979435,
    108.92475472163053,
    74.38953326406056,
    110.50162526355449,
    78.80020272137273,
    103.38549887401467,
    60.456178797087546],
    [172.45037522280097,
    174.18980860079296,
    171.85951120303966,
    171.25873283143545,
    160.75854520189236,
    169.48092881287826,
    161.08047079081524,
    43.790980696892355,
    155.81743843500544,
    148.34313468859568,
    50.064158066154825,
    158.17393868121383,
    155.45933597071283,
    104.13961612793646,
    122.48621995466567,
    12.659571151243236,
    155.47739976820696,
    36.35840136234119,
    141.46509251282993,
    52.22231034585671,
    123.93955712959554,
    48.70678496137736],
    [157.80029985177868,
    166.64538997712287,
    168.0831682477501,
    150.79185810187883,
    81.63338133068692,
    136.7688417886425,
    156.0427449565743,
    64.62014074943538,
    134.13409590741372,
    166.7144996483899,
    83.06487985111364,
    137.57855156387825,
    161.70988051925212,
    158.26751020335067,
    161.6145992638036,
    20.341342994222664,
    89.05568548228074,
    94.4404012109019,
    92.6710448391585,
    84.29466470232259,
    102.50198806831439,
    81.07344075971676],
    [164.61048487296463,
    166.69753879396632,
    151.47405450508515,
    168.5020909318047,
    177.63425236240934,
    174.7330268088124,
    145.4654379750723,
    42.86991708134899,
    153.08900088595007,
    134.8491293148981,
    43.73643136029382,
    137.91385384962447,
    141.99429988408673,
    35.29532617500614,
    149.73148092606309,
    39.58279269088692,
    97.81511357196189,
    69.62991820174915,
    118.9757196875293,
    64.44965128527988,
    118.95749101313618,
    56.8215987659198],
    [173.209905160051,
    157.16030492610872,
    165.79380466727238,
    123.34683269456526,
    136.655170969896,
    161.38274487765148,
    108.1508696557836,
    141.54845345507297,
    163.7819715847666,
    96.4445439032518,
    139.53903417978202,
    162.66455485477096,
    92.9698365113916,
    156.02305196123223,
    170.28264055871756,
    26.55822168376939,
    144.97755532945564,
    35.61079097040224,
    131.8075427639587,
    53.76514952601488,
    113.83370820754796,
    62.73407077223738],
    [147.81166415174795,
    138.75550606647207,
    173.83637797765573,
    160.53028376439065,
    167.1556975023265,
    175.19726577035362,
    169.3965806984324,
    165.1763236334311,
    175.03742960149515,
    166.8680517693352,
    174.1368371599068,
    175.9925822783681,
    164.59403380594682,
    175.92458489534212,
    177.8186869508598,
    35.537915354223976,
    82.34191603364215,
    97.31034606428267,
    103.13399997583348,
    80.7164738448546,
    115.20238261586425,
    67.52670889920101],
    [173.7196852879953,
    168.58514780096385,
    171.65435452552387,
    156.00702552495386,
    38.49656255705527,
    163.79639576358528,
    148.5960491201993,
    37.48555567413481,
    167.64209519897315,
    155.1755337784505,
    24.880016133312367,
    170.21155828612305,
    171.75548694580326,
    175.2421064202294,
    175.93253447339808,
    4.540502462278135,
    107.65095626830067,
    84.63403187839815,
    95.7970724026971,
    84.83732292633799,
    96.7165272152858,
    76.11077072204755],
    [172.4971884414722,
    163.50535095761128,
    161.3382135027539,
    160.9906137902653,
    58.1233387734657,
    145.53648691636485,
    163.42524699888492,
    31.067857240167527,
    161.16798305698052,
    167.70347960775857,
    9.449133536541291,
    173.52549477367612,
    175.11578049011501,
    23.998668825698605,
    173.7251972864031,
    12.013652891095097,
    72.25954253190999,
    112.50077900090913,
    91.92565665889654,
    90.34540333698749,
    107.0622984837979,
    64.36124855516867],
    [160.42070698232854,
    171.6326924622078,
    176.91404599640973,
    127.90502165543539,
    152.95313328286537,
    176.0835201041651,
    134.7189307970298,
    159.84521842637614,
    176.80248459727065,
    134.37565720287307,
    164.1263747307636,
    172.78583956329075,
    125.03660664654568,
    172.70091268821076,
    176.3104207260561,
    29.60556585831769,
    119.07731444146329,
    61.33550061597804,
    123.45285664809462,
    56.947194402081216,
    120.14588852473086,
    66.28184278329036],
    [167.90147670853221,
    147.44170044244433,
    122.24059874281565,
    157.44915247041325,
    54.57103898441296,
    151.16537571417965,
    162.1294178505243,
    49.920143235229666,
    155.27973386754664,
    168.6784260478335,
    44.06311798921394,
    151.81833882693962,
    171.384027507001,
    74.10235342001445,
    121.2032377756793,
    21.56656067128743,
    96.56424587325725,
    83.22658720620554,
    111.15047059616374,
    67.68729298595252,
    123.7894674422179,
    51.84289834200386],
    [158.37403484980118,
    137.67517573726525,
    127.9136189417235,
    177.75109422970135,
    175.72825921888077,
    178.29042485531448,
    175.04583567282322,
    171.4790938705805,
    175.15815082605255,
    172.59265307362088,
    163.8472963759443,
    177.24769017914772,
    158.3565880495092,
    35.47587995079752,
    162.62635984954426,
    14.654442696966491,
    114.80805422322739,
    79.5262614853159,
    125.9146210050736,
    64.75358989150241,
    136.24657711866305,
    33.97241493463713],
    [158.90594901451027,
    159.98551189106598,
    170.06163843008542,
    170.57638596498163,
    174.48794361274057,
    176.21484626205864,
    172.4392265645987,
    174.62173703279848,
    178.90218653552915,
    164.6578010392272,
    91.96808740328191,
    144.69660007024848,
    157.1310562760874,
    129.26057115175476,
    152.84712516673008,
    24.758785833927043,
    95.46061952465976,
    87.28947577949384,
    125.02459284514833,
    66.18325518416123,
    123.99475342690447,
    46.50869952326895]]



    import av
    from streamlit_webrtc import webrtc_streamer
    import math

   
    ff=[]

    def lii(li):
        j=0
        k=None
        m=None
        up=[]
        for i in li:
            if i==k:
                j+=1
                if j==2:
                    if m!=i:
                        up.append(i)
                    m=i
                    j=0
            k=i
        return up


    def ca(a, b, c):
        a_coords = (a.x, a.y, a.z)
        b_coords = (b.x, b.y, b.z)
        c_coords = (c.x, c.y, c.z)

        ba = [a_coords[i] - b_coords[i] for i in range(3)]
        bc = [c_coords[i] - b_coords[i] for i in range(3)]

        dot_product = sum(ba[i] * bc[i] for i in range(3))
        magnitude_ba = math.sqrt(sum(ba[i] ** 2 for i in range(3)))
        magnitude_bc = math.sqrt(sum(bc[i] ** 2 for i in range(3)))

        cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

        angle = math.acos(cosine_angle)
        angle = math.degrees(angle)

        return angle
    def find_angle(wrist0=0,thumb_cm1=0,thumb_mcp2=0,thumb_ip3=0,thumb_tip4=0,index_mcp5=0,index_pip6=0,index_dip7=0,index_tip8=0,middle_mcp9=0,middle_pip10=0,middle_dip11=0,middle_tip12=0,ring_mcp13=0,ring_pip14=0,ring_dip15=0,ring_tip16=0,pinky_mcp17=0,pinky_pip18=0,pinky_dip19=0,pinky_tip20=0):
        a012=ca(wrist0,thumb_cm1,thumb_mcp2)
        a123=ca(thumb_cm1,thumb_mcp2,thumb_ip3)
        a234=ca(thumb_mcp2,thumb_ip3,thumb_tip4)
        a056=ca(wrist0,index_mcp5,index_pip6)
        a567=ca(index_mcp5,index_pip6,index_dip7)
        a678=ca(index_pip6,index_dip7,index_tip8)
        a0910=ca(wrist0,middle_mcp9,middle_pip10)
        a91011=ca(middle_mcp9,middle_pip10,middle_dip11)
        a101112=ca(middle_pip10,middle_dip11,middle_tip12)
        a01314=ca(wrist0,ring_mcp13,ring_pip14)
        a131415=ca(ring_mcp13,ring_pip14,ring_dip15)
        a141516=ca(ring_pip14,ring_dip15,ring_tip16)
        a01718=ca(wrist0,pinky_mcp17,pinky_pip18)
        a171819=ca(pinky_mcp17,pinky_pip18,pinky_dip19)
        a181920=ca(pinky_pip18,pinky_dip19,pinky_tip20)
        a105=ca(thumb_cm1,wrist0,index_mcp5)
        a659=ca(index_pip6,index_mcp5,middle_mcp9)
        a5910=ca(index_mcp5,middle_mcp9,middle_pip10)
        a10913=ca(middle_pip10,middle_mcp9,ring_mcp13)
        a91314=ca(middle_mcp9,ring_mcp13,ring_pip14)
        a141317=ca(ring_pip14,ring_mcp13,pinky_mcp17)
        a131718=ca(ring_mcp13,pinky_mcp17,pinky_pip18)
        my=[a012,a123,a234,a056,a567,a678,a0910,a91011,a101112,a01314,a131415,a141516,a01718,a171819,a181920,a105,a659,a5910,a10913,a91314,a141317,a131718]
        return my


    def ssd(list1, list2):
        return sum((x - y) ** 2 for x, y in zip(list1, list2))

    def best(li):
        nn=10000000000000000000000000000000000000000000000000**25
        index=0
        for i in pp:
            SSD=ssd(li,i)
            if nn>SSD:
                good=i
                nn,SSD=SSD,nn
        for i, sublist in enumerate(pp):
                if sublist == good:
                    index = i
        return index

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
    pretex=None







    def callback(frame:av.VideoFrame) -> av.VideoFrame:
        frame_rgb=frame.to_ndarray(format='bgr24')
        results=hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist0 = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_tip4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_cm1= hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
                thumb_mcp2= hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                thumb_ip3= hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                index_mcp5= hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                index_pip6= hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                index_dip7= hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                middle_mcp9= hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                middle_pip10= hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                middle_dip11= hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                middle_tip12= hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_pip14= hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                ring_mcp13= hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                ring_dip15= hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
                ring_tip16= hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip20= hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                pinky_dip19= hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
                pinky_mcp17= hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                pinky_pip18= hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
            
            gj=find_angle(wrist0,thumb_cm1,thumb_mcp2,thumb_ip3,thumb_tip4,index_mcp5,index_pip6,index_dip7,index_tip8,middle_mcp9,middle_pip10,middle_dip11,middle_tip12,ring_mcp13,ring_pip14,ring_dip15,ring_tip16,pinky_mcp17,pinky_pip18,pinky_dip19,pinky_tip20)
            bb=['R','J','H','S','A','X','C','F','Y','V','G','K','O','T','D','M','L','Q','B','I','N','P','E','W','U']
            ind=best(gj)
            text = bb[ind]
            
            ff.append(text)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 255, 0) 
            thickness = 4
            org = (50, 50) 
            cv2.putText(frame_rgb, text, org, font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        
        return av.VideoFrame.from_ndarray(frame_rgb,format='bgr24')
    import logging
    import logging.handlers
    import threading

    import os
    from collections import deque
    from pathlib import Path
    import av
    import streamlit as st
    from twilio.rest import Client
    import cv2
    from streamlit_webrtc import  webrtc_streamer

    HERE = Path(__file__).parent

    logger = logging.getLogger(__name__)

    @st.cache_data  # type: ignore
    def get_ice_servers():
        try:
            account_sid = os.environ["TWILIO_ACCOUNT_SID"]
            auth_token = os.environ["TWILIO_AUTH_TOKEN"]
        except KeyError:
            logger.warning(
                "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
            )
            return [{"urls": ["stun:stun.l.google.com:19302"]}]

        client = Client(account_sid, auth_token)

        token = client.tokens.create()

        return token.ice_servers




    st.header("Real Time Sign-2-Letter")




    with_video_page = "Text IN Air"
    app_mode = st.selectbox("Choose ", [ with_video_page])


    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    if app_mode == with_video_page:
        




        frames_deque_lock = threading.Lock()
        frames_deque: deque = deque([])

    

        webrtc_ctx=webrtc_streamer(key='hi',video_frame_callback=callback,media_stream_constraints={"video": True, "audio": False})

        status_indicator = st.empty()

        if  webrtc_ctx.state.playing:
            

            stream=None
            while True:
                if webrtc_ctx.state.playing:
                    
                    status_indicator.write(lii(ff))



nav_selection = st.sidebar.radio("Navigation", ["Info", "Speach-2-Signs",'Text-2-Sign','Hand-Texting'])

# Render the selected page based on user's choice
if nav_selection == "Info":
    page_home()
elif nav_selection == "Speach-2-Signs":
    page_about()
elif nav_selection=='Text-2-Sign':
    mat()
elif nav_selection== 'Hand-Texting':
    bb()