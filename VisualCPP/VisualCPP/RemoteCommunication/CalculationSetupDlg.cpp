// CalculationSetupDlg.cpp : implementation file
//

#include "stdafx.h"

#include "RemoteCommunication.h"
#include "CalculationSetupDlg.h"

#include <CRColorimeter.h>


// CCalculationSetupDlg dialog

IMPLEMENT_DYNAMIC(CCalculationSetupDlg, CDialogEx)

CCalculationSetupDlg::CCalculationSetupDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CCalculationSetupDlg::IDD, pParent)
	, m_fWhitex(__CR_STEPZONE_LO_DEFAULT * 100.0F)
	, m_fWhitey(__CR_STEPZONE_HI_DEFAULT * 100.0F)
{
}

CCalculationSetupDlg::~CCalculationSetupDlg()
{
}

void CCalculationSetupDlg::SetColorimeter(CCRColorimeter* pColorimeter)
{
	m_pColorimeter = pColorimeter;
}

void CCalculationSetupDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
		
	DDX_Text(pDX, IDC_EDIT_WHITEX, m_fWhitex);
	DDX_Text(pDX, IDC_EDIT_WHITEY, m_fWhitey);

	
	DDX_Control(pDX, IDC_SPIN_WHITEX, m_whitexSpinButtonCtrl);
	DDX_Control(pDX, IDC_SPIN_WHITEY, m_whiteySpinButtonCtrl);

	DDX_Control(pDX, IDC_EDIT_WHITEX, m_whitexEdit);
	DDX_Control(pDX, IDC_EDIT_WHITEY, m_whiteyEdit);
}


BEGIN_MESSAGE_MAP(CCalculationSetupDlg, CDialogEx)

	ON_EN_CHANGE(IDC_EDIT_WHITEX, &CCalculationSetupDlg::OnEnChangeEditWhitex)
END_MESSAGE_MAP()


// CCalculationSetupDlg message handlers

BOOL CCalculationSetupDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	// Extra initialization here
		
	m_fWhitex = __CR_DOMINANTWAVELENGTH_WHITEX;
	m_fWhitey = __CR_DOMINANTWAVELENGTH_WHITEY;

	m_whitexSpinButtonCtrl.SetRange32(0, (int)__CR_DOMINANTWAVELENGTH_WHITEX * 10000.0F);
	m_whiteySpinButtonCtrl.SetRange32(0, (int)__CR_DOMINANTWAVELENGTH_WHITEY * 10000.0F);
	
	UpdateData(FALSE);

	return TRUE;
}

void CCalculationSetupDlg::EnableSettings(BOOL enable)
{
    m_whitexEdit.EnableWindow(enable);
    m_whiteyEdit.EnableWindow(enable);
}

void CCalculationSetupDlg::UpdateSettings(BOOL enable)
{
	if (m_pColorimeter)
    {

    }
}

void CCalculationSetupDlg::OnEnChangeEditWhitex()
{
	// TODO:  If this is a RICHEDIT control, the control will not
	// send this notification unless you override the CDialogEx::OnInitDialog()
	// function and call CRichEditCtrl().SetEventMask()
	// with the ENM_CHANGE flag ORed into the mask.

	// TODO:  Add your control notification handler code here
}
