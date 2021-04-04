function [erg, cent, crest, flat, oddeven] = pyTimbre(filepath)
%
%   Convenience wrapper for Timbre Toolbox use in Python.
%
%   This could certainly be extended. At the moment it accepts a filepath,
%   generates the default data for that file, and returns it to Python
%   somehow. TODO.
%

% Build descriptor preferences.
evalConfig = struct();
evalConfig.STFT = struct();
evalConfig.Harmonic = struct();

% Instruct TT to calculate the following.
evalConfig.STFT.FrameErg = struct();
evalConfig.STFT.SpecCent = struct();
evalConfig.STFT.SpecCrest = struct();
evalConfig.STFT.SpecFlat = struct();
evalConfig.Harmonic.OddEvenRatio = struct();

sound = SoundFile(filepath);
sound.Eval(evalConfig);

erg = sound.reps.STFT.descrs.FrameErg.value;
cent = sound.reps.STFT.descrs.SpecCent.value;
crest = sound.reps.STFT.descrs.SpecCrest.value;
flat = sound.reps.STFT.descrs.SpecFlat.value;
oddeven = sound.reps.Harmonic.descrs.OddEvenRatio.value;
end