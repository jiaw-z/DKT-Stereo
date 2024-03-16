from .pcvnet import PCVNet, sequence_loss_pcvnet
from .raft_stereo import RAFTStereo, sequence_loss_raft
from .nerf_stereo import ns_loss
from .cgi import CGI_Stereo, loss_cgi
from .gwcnet import GWCNet, loss_gwcnet

__models__ = {
    "RAFTStereo": RAFTStereo,
    "PCVNet": PCVNet,
    "CGI_Stereo": CGI_Stereo,
    "GWCNet": GWCNet,
}


__losses__ = {
    "sequence_loss_raft": sequence_loss_raft,
    "sequence_loss_pcvnet": sequence_loss_pcvnet,
    "ns_loss": ns_loss,
    "loss_cgi": loss_cgi,
    "loss_gwcnet": loss_gwcnet,
}
