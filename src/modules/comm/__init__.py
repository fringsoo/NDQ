from .information_bottleneck_comm import IBComm
from .information_bottleneck_comm_full import IBFComm
from .information_bottleneck_comm_not_IB import IBNIBComm
from .tar_comm import TarComm
from .information_bottleneck_pruned_comm import IBPComm
from .information_bottleneck_comm_full_new import IPComm_mind, IPComm_speaker, IPComm_listener, IPComm_speaker_simple, IPComm_listener_simple, IPComm_speaker_train, IPComm_listener_train

REGISTRY = {"information_bottleneck": IBComm,
            "information_bottleneck_full": IBFComm,
            "information_bottleneck_not_IB": IBNIBComm,
            "tar": TarComm,
            "information_bottleneck_pruned": IBPComm,
            "information_pragmatics": [IPComm_mind, IPComm_speaker, IPComm_listener, IPComm_speaker_simple, IPComm_listener_simple, IPComm_speaker_train, IPComm_listener_train]}
