// FlickerSetupDlg.cpp : implementation file
//

#include "stdafx.h"

#include "RemoteCommunication.h"
#include "FlickerSetupDlg.h"

#include <CRColorimeter.h>


// CFlickerSetupDlg dialog

IMPLEMENT_DYNAMIC(CFlickerSetupDlg, CDialogEx)

CFlickerSetupDlg::CFlickerSetupDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CFlickerSetupDlg::IDD, pParent)
	, m_bMaxFreqFlickerSearch(FALSE)
	, m_nFilterType(0)
	, m_nOrder(0)
	, m_fFrequency(0)
	, m_fBandwidth(0)
{

	m_fBandwidth = 0.0f;
	m_pColorimeter = NULL;
}

CFlickerSetupDlg::~CFlickerSetupDlg()
{
}

void CFlickerSetupDlg::SetColorimeter(CCRColorimeter* pColorimeter)
{
	m_pColorimeter = pColorimeter;
}

void CFlickerSetupDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_CBIndex(pDX, IDC_COMBO_FILTER_TYPE, m_nFilterType);
	DDX_Control(pDX, IDC_COMBO_FILTER_TYPE, m_filterTypeComboBox);
	DDX_Check(pDX, IDC_CHECK_MAX_FREQ_FLICKER_SEARCH, m_bMaxFreqFlickerSearch);
	DDX_Text(pDX, IDC_EDIT_MAX_FREQ_FLICKER_SEARCH, m_fMaxFreqFlickerSearch);
	DDX_Text(pDX, IDC_EDIT_ORDER, m_nOrder);
	DDX_Text(pDX, IDC_EDIT_FREQUENCY, m_fFrequency);
	DDX_Text(pDX, IDC_EDIT_BANDWIDTH, m_fBandwidth);
	DDV_MinMaxFloat(pDX, m_fMaxFreqFlickerSearch, __CR_MAX_SEARCH_FREQUENCY_MIN, __CR_MAX_SEARCH_FREQUENCY_MAX);
	DDV_MinMaxFloat(pDX, m_fFrequency, __CR_FILTER_FREQUENCY_MIN, __CR_FILTER_FREQUENCY_MAX);
	DDV_MinMaxFloat(pDX, m_fBandwidth, __CR_FILTER_BANDWIDTH_MIN, __CR_FILTER_BANDWIDTH_MAX);
	DDV_MinMaxInt(pDX, m_nOrder, __CR_FILTER_ORDER_MIN, __CR_FILTER_ORDER_MAX);
	DDX_Control(pDX, IDC_SPIN_MAX_FREQ_FLICKER_SEARCH, m_maxFreqFlickerSearchSpinButtonCtrl);
	DDX_Control(pDX, IDC_SPIN_BANDWIDTH, m_bandwidthSpinButtonCtrl);
	DDX_Control(pDX, IDC_SPIN_FREQUENCY, m_frequencySpinButtonCtrl);
	DDX_Control(pDX, IDC_SPIN_ORDER, m_orderSpinButtonCtrl);
	DDX_Control(pDX, IDC_CHECK_MAX_FREQ_FLICKER_SEARCH, m_maxFreqFlickerSearchCheckBox);
	DDX_Control(pDX, IDC_EDIT_MAX_FREQ_FLICKER_SEARCH, m_maxFreqFlickerSearchEdit);
	DDX_Control(pDX, IDC_EDIT_ORDER, m_orderEdit);
	DDX_Control(pDX, IDC_EDIT_FREQUENCY, m_frequencyEdit);
	DDX_Control(pDX, IDC_EDIT_BANDWIDTH, m_bandwidthEdit);
}


BEGIN_MESSAGE_MAP(CFlickerSetupDlg, CDialogEx)
END_MESSAGE_MAP()


// CFlickerSetupDlg message handlers

BOOL CFlickerSetupDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	// Extra initialization here
		
	m_filterTypeComboBox.AddString(_T("None"));
	m_filterTypeComboBox.AddString(_T("Low Pass"));
	m_filterTypeComboBox.AddString(_T("High Pass"));
	m_filterTypeComboBox.AddString(_T("Band Pass"));
	m_filterTypeComboBox.AddString(_T("Band Stop"));
	
	m_nFilterType = 0;
	m_nOrder = __CR_FILTER_ORDER_MIN;
	m_fBandwidth = __CR_FILTER_BANDWIDTH_MAX;
	m_fFrequency = __CR_FILTER_FREQUENCY_MAX;
	m_fMaxFreqFlickerSearch = __CR_MAX_SEARCH_FREQUENCY_DEFAULT;
	
	m_orderSpinButtonCtrl.SetRange32(__CR_FILTER_ORDER_MIN, __CR_FILTER_ORDER_MAX);
	m_bandwidthSpinButtonCtrl.SetRange32((int)__CR_FILTER_BANDWIDTH_MIN, (int)__CR_FILTER_BANDWIDTH_MAX);
	m_frequencySpinButtonCtrl.SetRange32((int)__CR_FILTER_FREQUENCY_MIN, (int)__CR_FILTER_FREQUENCY_MAX);
	m_maxFreqFlickerSearchSpinButtonCtrl.SetRange32((int)__CR_MAX_SEARCH_FREQUENCY_MIN, (int)__CR_MAX_SEARCH_FREQUENCY_MAX);

	UpdateData(FALSE);

	return TRUE;
}

void CFlickerSetupDlg::EnableSettings(BOOL enable)
{
	m_filterTypeComboBox.EnableWindow(enable);
    m_maxFreqFlickerSearchCheckBox.EnableWindow(enable);
    m_maxFreqFlickerSearchEdit.EnableWindow(enable);
    m_orderEdit.EnableWindow(enable);
    m_frequencyEdit.EnableWindow(enable);
    m_bandwidthEdit.EnableWindow(enable);
}

void CFlickerSetupDlg::UpdateSettings(BOOL enable)
{
	if (m_pColorimeter)
    {
        //m_checkBoxMaxFreqFlickerSearch.SetChecked(m_pColorimeter->FlickerMaxSearchFrequencyEnabled());

		CString strValue;
		strValue.Format(_T("%f"), m_pColorimeter->FlickerMaxSearchFrequency());
		m_maxFreqFlickerSearchEdit.SetWindowText(strValue);
		m_filterTypeComboBox.SetCurSel(m_pColorimeter->FlickerFilterType());
		strValue.Format(_T("%d"), m_pColorimeter->FlickerFilterOrder());
		m_orderEdit.SetWindowText(strValue);
		strValue.Format(_T("%f"), m_pColorimeter->FlickerFilterFrequency());
		m_frequencyEdit.SetWindowText(strValue);
		strValue.Format(_T("%f"), m_pColorimeter->FlickerFilterBandwidth());
		m_bandwidthEdit.SetWindowText(strValue);
    }
}