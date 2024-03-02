
from styletts2 import librispeech,compute_style
# import pprint
# pprint.pprint(sys.path)
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

StyleTTS2_CONFIG_FILE="resources/models/styletts2/Models/LibriTTS/config.yml"
StyleTTS2_MODEL_FILE="resources/models/styletts2/Models/LibriTTS/epochs_2nd_00020.pth"

ref_s1 = compute_style("resources/models/styletts2/reference_audio/Gavin.wav")
ref_s2 = compute_style("resources/models/styletts2/reference_audio/Jane.wav")
# ref_s3 = compute_style("resources/models/styletts2/reference_audio/Me1.wav")
# ref_s4 = compute_style("resources/models/styletts2/reference_audio/Me2.wav")
# ref_s5 = compute_style("resources/models/styletts2/reference_audio/Me3.wav")
# ref_s6 = compute_style("resources/models/styletts2/reference_audio/Vinay.wav")
# ref_s7 = compute_style("resources/models/styletts2/reference_audio/Nima.wav")
# ref_s8 = compute_style("resources/models/styletts2/reference_audio/Yinghao.wav")
# ref_s9 = compute_style("resources/models/styletts2/reference_audio/Keith.wav")
# ref_s10 = compute_style("resources/models/styletts2/reference_audio/May.wav")
# ref_s11 = compute_style("resources/models/styletts2/reference_audio/June.wav")

"""
"Sure! Here's a list of some major cities in Canada:\n\n1. Toronto - Ontario\n2. Montreal - Quebec\n3. Vancouver - British Columbia\n4. Calgary - Alberta\n5. Edmonton - Alberta\n6. Ottawa - Ontario (the capital city)\n7. Winnipeg - Manitoba\n8. Halifax - Nova Scotia\n9. Kitchener-Waterloo - Ontario (technically two cities, but often referred to together)\n10. Regina - Saskatchewan\n11. Sherbrooke - Quebec\n12. London - Ontario\n13. Saint John - New Brunswick\n14. Brampton - Ontario\n15. Burlington - Ontario\n\nI hope that helps! Let me know if you have any other questions."), ("Sure! Here's a list of some major cities in Canada:\n\n1. Toronto - Ontario\n2. Montreal - Quebec\n3. Vancouver - British Columbia\n4. Calgary - Alberta\n5. Edmonton - Alberta\n6. Ottawa - Ontario (the capital city)\n7. Winnipeg - Manitoba\n8. Halifax - Nova Scotia\n9. Kitchener-Waterloo - Ontario (technically two cities, but often referred to together)\n10. Regina - Saskatchewan\n11. Sherbrooke - Quebec\n12. London - Ontario\n13. Saint John - New Brunswick\n14. Brampton - Ontario\n15. Burlington - Ontario\n\nI hope that helps! Let me know if you have any other questions.", "sure here's to montreal. columbia for oh.
"""
text = "Yea, his honourable worship is within, \nbut he hath a godly minister or two with him, and likewise a leech."
#librispeech(text=text,compute_style=ref_s2, voice='Jane',alpha=0.3, beta=0.5, diffusion_steps=10)
#librispeech(text=text,compute_style=ref_s11, voice='Jane',alpha=0.3, beta=0.7, diffusion_steps=10)

#from styletts2 import  ljspeech
#ljspeech(text=text,device="cuda")
