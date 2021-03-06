"""
Convenience macros for generating stimuli as per thesis.

See `synthesis.py` for condition descriptions.
"""

from src.defaults import SAMPLE_RATE, PITCH_RATE
from src.synthesis import EnvelopeMorpher, StimulusGenerator

# Instantiate one generator for all stimuli.
generator = StimulusGenerator(
    sr=SAMPLE_RATE,
    pr=PITCH_RATE,
    random_rate_lower_limit=4.,
    random_rate_upper_limit=12.,
)


def make_basic(args):
    return generator(
            f0=args['f0'],
            fm_depth=args['fm_depth'],
            env=args['env'],
            num_partials=args['num_partials'],
            length=args['length'],
            mod_rate=args['mod_rate'],
            mod_hold=args['mod_hold'],
            mod_fade=args['mod_fade'],
            audio_fade=args['audio_fade'],
    )


def make_frozen(args):
    return generator(
            f0=args['f0'],
            fm_depth=0.,
            env=args['env'],
            num_partials=args['num_partials'],
            length=args['length'],
            mod_rate=args['mod_rate'],
            mod_hold=args['mod_hold'],
            mod_fade=args['mod_fade'],
            audio_fade=args['audio_fade'],
    )


def make_shuffle(args, log):
    morpher = EnvelopeMorpher(args['env'], args['f0'])
    morpher.shuffle_phase(num_shifts=4)

    x = generator(
        f0=args['f0'],
        fm_depth=0.,
        env=morpher(),
        num_partials=args['num_partials'],
        length=args['length'],
        mod_rate=args['mod_rate'],
        mod_hold=args['mod_hold'],
        mod_fade=args['mod_fade'],
        audio_fade=args['audio_fade'],
    )

    x_raf = generator(
        f0=args['f0'],
        fm_depth=0.,
        env=morpher(),
        num_partials=args['num_partials'],
        length=args['length'],
        mod_rate=args['mod_rate'],
        mod_hold=args['mod_hold'],
        mod_fade=args['mod_fade'],
        audio_fade=args['audio_fade'],
        synth_mode='raf',
    )

    log.write("\nSHUFFLE\n" + "_"*7 + "\n")
    log.write(f"\n{morpher._log}\n")

    return x, x_raf


def make_simple(args, log):
    morpher = EnvelopeMorpher(args['env'], args['f0'])
    morpher.rap()
    morpher.shuffle_phase(num_shifts=4)

    x = generator(
        f0=args['f0'],
        fm_depth=0.,
        env=morpher(),
        num_partials=args['num_partials'],
        length=args['length'],
        mod_rate=args['mod_rate'],
        mod_hold=args['mod_hold'],
        mod_fade=args['mod_fade'],
        audio_fade=args['audio_fade'],
    )

    x_raf = generator(
        f0=args['f0'],
        fm_depth=0.,
        env=morpher(),
        num_partials=args['num_partials'],
        length=args['length'],
        mod_rate=args['mod_rate'],
        mod_hold=args['mod_hold'],
        mod_fade=args['mod_fade'],
        audio_fade=args['audio_fade'],
        synth_mode='raf',
    )

    log.write("\nSIMPLE\n" + "_" * 7 + "\n")
    log.write(f"\n{morpher._log}\n")

    return x, x_raf


def make_rag(args, log):
    morpher = EnvelopeMorpher(args['env'], args['f0'])
    morpher.rap(max_random_gain=10)
    morpher.shuffle_phase(num_shifts=4)

    x = generator(
        f0=args['f0'],
        fm_depth=0.,
        env=morpher(),
        num_partials=args['num_partials'],
        length=args['length'],
        mod_rate=args['mod_rate'],
        mod_hold=args['mod_hold'],
        mod_fade=args['mod_fade'],
        audio_fade=args['audio_fade'],
    )

    x_raf = generator(
        f0=args['f0'],
        fm_depth=0.,
        env=morpher(),
        num_partials=args['num_partials'],
        length=args['length'],
        mod_rate=args['mod_rate'],
        mod_hold=args['mod_hold'],
        mod_fade=args['mod_fade'],
        audio_fade=args['audio_fade'],
        synth_mode='raf'
    )

    log.write("\nRAG\n" + "_" * 7 + "\n")
    log.write(f"\n{morpher._log}\n")

    return x, x_raf


def make_fm_only(args):
    morpher = EnvelopeMorpher(args['env'], args['f0'])
    morpher.time_average()

    return generator(
        f0=args['f0'],
        fm_depth=args['fm_depth'],
        env=morpher(),
        num_partials=args['num_partials'],
        length=args['length'],
        mod_rate=args['mod_rate'],
        mod_hold=args['mod_hold'],
        mod_fade=args['mod_fade'],
        audio_fade=args['audio_fade'],
        synth_mode='default',
    )


def make_control(args):
    return generator(
        f0=args['f0'],
        fm_depth=0.,
        env=args['env'],
        num_partials=args['num_partials'],
        length=args['length'],
        mod_rate=args['mod_rate'],
        mod_hold=args['length'],
        mod_fade=0.,
        audio_fade=args['audio_fade'],
        synth_mode='pam',
    )


def make_pam(args):
    return generator(
        f0=args['f0'],
        fm_depth=0.,
        env=args['env'],
        num_partials=args['num_partials'],
        length=args['length'],
        mod_rate=args['mod_rate'],
        mod_hold=args['mod_hold'],
        mod_fade=args['mod_fade'],
        audio_fade=args['audio_fade'],
        synth_mode='pam',
    )
