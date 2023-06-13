# import numpy as np
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

# from fmriprep import config
from fmriprep.config import DEFAULT_MEMORY_MIN_GB
from fmriprep.interfaces import DerivativesDataSink


def init_bold_minimal_wf(
    # bids_root,
    # metadata,
    output_dir,
    name='bold_minimal_wf',
):
    """
    Set up a battery of datasinks to store derivatives in the right location.

    Parameters
    ----------
    output_dir : :obj:`str`
        Where derivatives should be written out to.
    name : :obj:`str`
        This workflow's identifier (default: ``func_derivatives_wf``).

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    # from niworkflows.interfaces.utility import KeySelect
    # from smriprep.workflows.outputs import _bids_relative

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=[
            't1w2fsnative_xfm',
            'template',
            'anat2std_xfm',
            'bold_file',
            't1w_mask',
            't1w_aseg',
            't1w_aparc',
            'bold_mask',
            'itk_bold_to_t1',
            't1w_brain',
            'ref_bold_mask',
            'ref_bold_brain',
            'bold_t1',
            'bold_split',
            'fieldwarp',
            'hmc_xforms',
        ]),
        name='inputnode')

    t1w2fsnative_xfm = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, desc='t1w2fsnative',
            dismiss_entities=("echo",),
            mode='image', suffix='xfm',
            extension='.txt',
            **{'from': 't1w', 'to': 'fsnative'},
        ),
        name='t1w2fsnative_xfm',
        run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, t1w2fsnative_xfm, [
            ('t1w2fsnative_xfm', 'in_file'),
            ('bold_file', 'source_file')]),
    ])

    anat2std_xfm = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, desc='anat2std',
            dismiss_entities=("echo",),
            suffix='xfm',
            mode='image', extension='.h5',
            **{'from': 't1w', 'to': 'std'},
        ),
        name='anat2std_xfm',
        run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, anat2std_xfm, [
            ('anat2std_xfm', 'in_file'),
            ('bold_file', 'source_file')]),
    ])

    itk_bold_to_t1 = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=("echo",),
            suffix='xfm',
            # mode='image',
            extension='.txt',
            **{'from': 'ref', 'to': 't1w'},
        ),
        name='itk_bold_to_t1',
        run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, itk_bold_to_t1, [
            ('itk_bold_to_t1', 'in_file'),
            ('bold_file', 'source_file')]),
    ])

    fieldwarp = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, desc='fieldwarp',
            dismiss_entities=("echo",),
            mode='image', extension='.nii.gz',
        ),
        name='fieldwarp',
        run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, fieldwarp, [
            ('fieldwarp', 'in_file'),
            ('bold_file', 'source_file')]),
    ])

    hmc_xforms = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, desc='hmc', suffix='xfm',
            dismiss_entities=("echo",),
            mode='image',
            extension='.txt',
            **{'from': 'bold', 'to': 'ref'},
        ),
        name='hmc_xforms',
        run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, hmc_xforms, [
            ('hmc_xforms', 'in_file'),
            ('bold_file', 'source_file')]),
    ])

    bold_t1 = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, desc='t1space', suffix='bold',
            dismiss_entities=("echo",),
            mode='image', extension='.nii.gz'),
        name='bold_t1',
        run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, bold_t1, [
            ('bold_t1', 'in_file'),
            ('bold_file', 'source_file')]),
    ])

    # bold_split
    def rename_split_func(in_file, source_file):
        import os
        basename = os.path.basename(in_file)
        assert basename.startswith('vol')
        assert basename.endswith('.nii.gz')
        vol = int(basename[3:-7])
        a, b = source_file.rsplit('_', 1)
        out_fn = f'{a}_desc-vol{vol:04d}_{b}.nii.gz'
        return out_fn

    rename_split = pe.MapNode(
        niu.Function(
            function=rename_split_func, input_names=['in_file', 'source_file'],
            output_names=['out_file']),
        name='rename_split',
        iterfield=['in_file'],
        run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)

    bold_split = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir, suffix='bold',
            dismiss_entities=("echo",),
            mode='image', extension='.nii.gz'),
        name='bold_split',
        iterfield=['in_file', 'source_file'],
        run_without_submitting=True, mem_gb=DEFAULT_MEMORY_MIN_GB)
    workflow.connect([
        (inputnode, rename_split, [
            ('bold_split', 'in_file'),
            ('bold_file', 'source_file')]),
        (rename_split, bold_split, [
            # ('in_file', 'in_file'),
            ('out_file', 'source_file')]),
        (inputnode, bold_split, [
            ('bold_split', 'in_file')]),
    ])

    return workflow
