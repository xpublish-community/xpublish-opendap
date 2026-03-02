"""DAP4 DSR (Dataset Services Response) and error XML generators."""

from __future__ import annotations

import xml.etree.ElementTree as ET

DAP4_NS = 'http://xml.opendap.org/ns/DAP/4.0#'


def generate_dsr(dataset_name: str, base_url: str) -> str:
    """Generate a DAP4 Dataset Services Response (DSR) XML document.

    The DSR describes the available services and their access URLs for a dataset.

    Args:
        dataset_name: The name of the dataset.
        base_url: The base URL for constructing service endpoints.

    Returns:
        XML string of the DSR document.
    """
    root = ET.Element('DatasetServices')
    root.set('xmlns', DAP4_NS)
    root.set('xml:base', base_url)
    root.set('name', dataset_name)
    root.set('dapVersion', '4.0')

    # DMR service
    dmr_svc = ET.SubElement(root, 'Service')
    dmr_svc.set('title', 'Dataset Metadata Response')
    dmr_svc.set('role', 'http://services.opendap.org/dap4/dataset-metadata')
    link = ET.SubElement(dmr_svc, 'Link')
    link.set('href', f'{base_url}.dmr')
    link.set('type', 'application/vnd.opendap.dap4.dataset-metadata+xml')

    # DAP (data) service
    dap_svc = ET.SubElement(root, 'Service')
    dap_svc.set('title', 'DAP4 Data Response')
    dap_svc.set('role', 'http://services.opendap.org/dap4/data')
    link = ET.SubElement(dap_svc, 'Link')
    link.set('href', f'{base_url}.dap')
    link.set('type', 'application/vnd.opendap.dap4.data')

    # DAP2 compatibility services
    dds_svc = ET.SubElement(root, 'Service')
    dds_svc.set('title', 'DAP2 DDS')
    dds_svc.set('role', 'http://services.opendap.org/dap2/dds')
    link = ET.SubElement(dds_svc, 'Link')
    link.set('href', f'{base_url}.dds')
    link.set('type', 'text/plain')

    das_svc = ET.SubElement(root, 'Service')
    das_svc.set('title', 'DAP2 DAS')
    das_svc.set('role', 'http://services.opendap.org/dap2/das')
    link = ET.SubElement(das_svc, 'Link')
    link.set('href', f'{base_url}.das')
    link.set('type', 'text/plain')

    dods_svc = ET.SubElement(root, 'Service')
    dods_svc.set('title', 'DAP2 Data')
    dods_svc.set('role', 'http://services.opendap.org/dap2/data')
    link = ET.SubElement(dods_svc, 'Link')
    link.set('href', f'{base_url}.dods')
    link.set('type', 'application/octet-stream')

    ET.indent(root, space='  ')
    return ET.tostring(root, encoding='unicode', xml_declaration=True) + '\n'


def generate_dap4_error(code: int, message: str, http_code: int = 400) -> str:
    """Generate a DAP4 error XML document.

    Args:
        code: The DAP error code.
        message: The error message.
        http_code: The HTTP status code.

    Returns:
        XML string of the error document.
    """
    root = ET.Element('Error')
    root.set('xmlns', DAP4_NS)
    root.set('httpcode', str(http_code))

    code_el = ET.SubElement(root, 'ErrorCode')
    code_el.text = str(code)

    msg_el = ET.SubElement(root, 'Message')
    msg_el.text = message

    ET.indent(root, space='  ')
    return ET.tostring(root, encoding='unicode', xml_declaration=True) + '\n'
